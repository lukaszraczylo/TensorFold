//! Inkling 2 Mini's loop kernels (inkling2/loop.cu): the image's module and its functions.
const std = @import("std");
const cuda = @import("cuda");
const names = @import("names.zig");
const plan = @import("plan.zig");

/// Threads a row's quantiser block runs; loop.cu's kThreads.
pub const actq_threads = 256;
pub const nvfp4_block = 16;
pub const mxfp8_block = 32;
/// Columns a block of the expert GEMVs computes (loop.cu's kExWarps: a warp a column).
pub const expert_tile = 8;
/// Experts a row picks.
pub const expert_slots = 8;
const elementwise_threads = 256;
/// Hidden width and controller rank of the loop kernels; loop.cu's kLoopMaxC and kLoopMaxRank bound them.
pub const gate_rank = 64;
pub const n_streams = 4;
/// The streams in kernel order: attn, k, v, mlp.
pub const stream_names = names.controller_streams;
const loop_hidden = 2304;
const msg_sources = 2;
/// ref.model._message_queries: the queries are raw in memory and the kernel multiplies each by D^-0.5 once.
const msg_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, loop_hidden)));

pub const Kernels = struct {
    mod: cuda.Module,
    fill: cuda.Function,
    actq_nvfp4: cuda.Function,
    actq_mxfp8: cuda.Function,
    actq_nvfp4_codes: cuda.Function,
    actq_mxfp8_codes: cuda.Function,
    ex_gu_nvfp4: cuda.Function,
    ex_gu_mxfp8: cuda.Function,
    ex_down_nvfp4: cuda.Function,
    ex_down_mxfp8: cuda.Function,
    situ: cuda.Function,
    ex_sum: cuda.Function,
    msg: cuda.Function,
    ctl_down: cuda.Function,
    ctl_blend: cuda.Function,
    ablit: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .fill = try mod.function("tf_ink2l_fill"),
            .actq_nvfp4 = try mod.function("tf_ink2l_actq_nvfp4"),
            .actq_mxfp8 = try mod.function("tf_ink2l_actq_mxfp8"),
            .actq_nvfp4_codes = try mod.function("tf_ink2l_actq_nvfp4_codes"),
            .actq_mxfp8_codes = try mod.function("tf_ink2l_actq_mxfp8_codes"),
            .ex_gu_nvfp4 = try mod.function("tf_ink2l_ex_gu_nvfp4"),
            .ex_gu_mxfp8 = try mod.function("tf_ink2l_ex_gu_mxfp8"),
            .ex_down_nvfp4 = try mod.function("tf_ink2l_ex_down_nvfp4"),
            .ex_down_mxfp8 = try mod.function("tf_ink2l_ex_down_mxfp8"),
            .situ = try mod.function("tf_ink2l_situ"),
            .ex_sum = try mod.function("tf_ink2l_ex_sum"),
            .msg = try mod.function("tf_ink2l_msg"),
            .ctl_down = try mod.function("tf_ink2l_ctl_down"),
            .ctl_blend = try mod.function("tf_ink2l_ctl_blend"),
            .ablit = try mod.function("tf_ink2l_ablit"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// `extra` are the codes kernels' output pointers, between `out` and the sizes.
fn actq(f: cuda.Function, block: usize, s: cuda.Stream, x: u64, out: u64, extra: []const u64, rows: usize, kdim: usize) !void {
    if (kdim == 0 or kdim % block != 0) return error.BadWidth;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(out);
    for (extra) |p| args.add(p);
    args.add(@as(i32, @intCast(rows)));
    args.add(@as(i32, @intCast(kdim)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = actq_threads } }, s, &args);
}

/// `x` is bf16 [rows, kdim], `out` float32 [rows, kdim]: per-row NVFP4 with the four-over-six rule, dequantised.
pub fn actNvfp4(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_nvfp4, nvfp4_block, s, x, out, &.{}, rows, kdim);
}

/// Same shapes as `actNvfp4`: per-32 MXFP8 with power-of-two scales.
pub fn actMxfp8(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_mxfp8, mxfp8_block, s, x, out, &.{}, rows, kdim);
}

/// As `actNvfp4`, plus ref.quant.nvfp4_codes: e2m1 values float32 [rows, kdim], e4m3 block scales float32 [rows, kdim/16], row amax float32 [rows].
pub fn actNvfp4Codes(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, values: u64, scales: u64, amax: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_nvfp4_codes, nvfp4_block, s, x, out, &.{ values, scales, amax }, rows, kdim);
}

/// As `actMxfp8`, plus ref.quant.mxfp8_codes: e4m3 values float32 [rows, kdim] and UE8M0 biased exponents u8 [rows, kdim/32].
pub fn actMxfp8Codes(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, values: u64, exps: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_mxfp8_codes, mxfp8_block, s, x, out, &.{ values, exps }, rows, kdim);
}

/// One layer's packed expert tensors of one GEMM, as stored; `amax` and `outer` are 0 for MXFP8.
pub const ExpertGemm = struct { w: u64, scale: u64, amax: u64 = 0, outer: u64 = 0 };

fn expertGemm(f: cuda.Function, s: cuda.Stream, act: u64, pick: u64, g: ExpertGemm, out: u64, rows: usize, n: usize, k: usize) !void {
    const pairs = rows * expert_slots;
    if (n == 0 or k == 0 or k % mxfp8_block != 0) return error.BadWidth;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    for ([_]u64{ act, pick, g.w, g.scale, g.amax, g.outer, out }) |v| args.add(v);
    for ([_]usize{ pairs, n, k }) |v| args.add(@as(i32, @intCast(v)));
    const tiles = (n + expert_tile - 1) / expert_tile;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(pairs), .y = @intCast(tiles) }, .block = .{ .x = actq_threads } }, s, &args);
}

/// act float32 [rows, k] (quantised), pick int32 [rows, 8]; gu bf16 [rows * 8, n] = bf16(act . w13[pick]) over `k`.
pub fn expertGateUp(k: *const Kernels, s: cuda.Stream, prec: plan.Precision, act: u64, pick: u64, g: ExpertGemm, gu: u64, rows: usize, n: usize, kdim: usize) !void {
    return expertGemm(if (prec == .nvfp4) k.ex_gu_nvfp4 else k.ex_gu_mxfp8, s, act, pick, g, gu, rows, n, kdim);
}

/// act float32 [rows * 8, k] (quantised SwiGLU output); dn float32 [rows * 8, n] holds bf16-rounded values.
pub fn expertDown(k: *const Kernels, s: cuda.Stream, prec: plan.Precision, act: u64, pick: u64, g: ExpertGemm, dn: u64, rows: usize, n: usize, kdim: usize) !void {
    return expertGemm(if (prec == .nvfp4) k.ex_down_nvfp4 else k.ex_down_mxfp8, s, act, pick, g, dn, rows, n, kdim);
}

/// ref.ops.situ over rows * 8 pairs: gu bf16 [pairs, 2 * inter], gamma float32 [pairs], y bf16 [pairs, inter].
pub fn situ(k: *const Kernels, s: cuda.Stream, gu: u64, gamma: u64, y: u64, rows: usize, inter: usize, gate_cap: f32, up_cap: f32) !void {
    if (rows == 0) return;
    const pairs = rows * expert_slots;
    var args: cuda.Args = .{};
    for ([_]u64{ gu, gamma, y }) |v| args.add(v);
    args.add(@as(i32, @intCast(pairs)));
    args.add(@as(i32, @intCast(inter)));
    args.add(gate_cap);
    args.add(up_cap);
    const blocks = (pairs * inter + elementwise_threads - 1) / elementwise_threads;
    try cuda.launch.launch(k.situ, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = elementwise_threads } }, s, &args);
}

/// dn float32 [rows * 8, n] summed over the 8 slots in slot order into bf16 out [rows, n].
pub fn expertSum(k: *const Kernels, s: cuda.Stream, dn: u64, out: u64, rows: usize, n: usize) !void {
    if (rows == 0) return;
    var args: cuda.Args = .{};
    args.add(dn);
    args.add(out);
    args.add(@as(i32, @intCast(rows)));
    args.add(@as(i32, @intCast(n)));
    const blocks = (rows * n + elementwise_threads - 1) / elementwise_threads;
    try cuda.launch.launch(k.ex_sum, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = elementwise_threads } }, s, &args);
}

/// Cross-pass messages of core layer `m`: bank = the previous pass's two blocks (bf16 [rows, 2304] each), qa and qkv =
/// the raw cross-pass query tensors (float32 [24, 2304]), out float32 [4, rows, 2304] in stream order attn, k, v, mlp.
pub fn messages(k: *const Kernels, s: cuda.Stream, bank: [msg_sources]u64, qa: u64, qkv: u64, m: u32, out: u64, rows: usize) !void {
    if (rows == 0) return;
    var args: cuda.Args = .{};
    for (bank) |p| args.add(p);
    args.add(qa);
    args.add(qkv);
    args.add(@as(i32, @intCast(m)));
    args.add(msg_scale);
    args.add(out);
    args.add(@as(i32, @intCast(rows)));
    args.add(@as(i32, loop_hidden));
    try cuda.launch.launch(k.msg, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = actq_threads } }, s, &args);
}

/// One stream of one core layer: dest and msg float32 [rows, 2304], the controller's bf16 weights, the sublayer norm
/// weight bf16 [2304]; h bf16 [rows, 64] is scratch and out bf16 [rows, 2304] the normed blend.
pub const ControllerArgs = struct { dest: u64, msg: u64, down: u64, gain: u64, forget: u64, norm: u64, h: u64, out: u64, rows: usize };

pub fn controller(k: *const Kernels, s: cuda.Stream, a: ControllerArgs) !void {
    if (a.rows == 0) return;
    const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(a.rows) }, .block = .{ .x = actq_threads } };
    var down: cuda.Args = .{};
    for ([_]u64{ a.dest, a.msg, a.down, a.h }) |p| down.add(p);
    for ([_]usize{ a.rows, loop_hidden, gate_rank }) |v| down.add(@as(i32, @intCast(v)));
    try cuda.launch.launch(k.ctl_down, cfg, s, &down);
    var blend: cuda.Args = .{};
    for ([_]u64{ a.dest, a.msg, a.h, a.gain, a.forget, a.norm, a.out }) |p| blend.add(p);
    for ([_]usize{ a.rows, loop_hidden, gate_rank }) |v| blend.add(@as(i32, @intCast(v)));
    try cuda.launch.launch(k.ctl_blend, cfg, s, &blend);
}

const Fake = struct {
    var grid_x: u32 = 0;
    var block_x: u32 = 0;
    var which: usize = 0;
    var count: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        grid_x = gx;
        block_x = bx;
        which = @intFromPtr(f);
        count += 1;
        return 0;
    }
};

test "actq launches one block a row and refuses a width off the block" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const fun = struct {
        fn of(drv: *const cuda.Driver, i: usize) cuda.Function {
            return .{ .d = drv, .handle = @ptrFromInt(i) };
        }
    };
    const k: Kernels = .{ .mod = .{ .d = &d, .handle = null }, .fill = fun.of(&d, 1), .actq_nvfp4 = fun.of(&d, 2), .actq_mxfp8 = fun.of(&d, 3), .actq_nvfp4_codes = fun.of(&d, 4), .actq_mxfp8_codes = fun.of(&d, 5), .ex_gu_nvfp4 = fun.of(&d, 6), .ex_gu_mxfp8 = fun.of(&d, 7), .ex_down_nvfp4 = fun.of(&d, 8), .ex_down_mxfp8 = fun.of(&d, 9), .situ = fun.of(&d, 10), .ex_sum = fun.of(&d, 11), .msg = fun.of(&d, 12), .ctl_down = fun.of(&d, 13), .ctl_blend = fun.of(&d, 14), .ablit = fun.of(&d, 15) };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try actNvfp4(&k, s, 0x1000, 0x2000, 64, 2304);
    try std.testing.expectEqual(@as(u32, 64), Fake.grid_x);
    try std.testing.expectEqual(@as(u32, actq_threads), Fake.block_x);
    try std.testing.expectEqual(@as(usize, 2), Fake.which);
    try actMxfp8(&k, s, 0x1000, 0x2000, 3, 1152);
    try std.testing.expectEqual(@as(u32, 3), Fake.grid_x);
    try std.testing.expectEqual(@as(usize, 3), Fake.which);
    const launched = Fake.count;
    try std.testing.expectError(error.BadWidth, actNvfp4(&k, s, 0, 0, 1, 2312));
    try std.testing.expectError(error.BadWidth, actMxfp8(&k, s, 0, 0, 1, 1136));
    try std.testing.expectError(error.BadWidth, actNvfp4(&k, s, 0, 0, 1, 0));
    try std.testing.expectEqual(launched, Fake.count);
    try actNvfp4Codes(&k, s, 0, 0, 0, 0, 0, 5, 2304);
    try std.testing.expectEqual(@as(usize, 4), Fake.which);
    try std.testing.expectEqual(@as(u32, 5), Fake.grid_x);
    try actMxfp8Codes(&k, s, 0, 0, 0, 0, 7, 1152);
    try std.testing.expectEqual(@as(usize, 5), Fake.which);
    try std.testing.expectError(error.BadWidth, actMxfp8Codes(&k, s, 0, 0, 0, 0, 1, 1136));
    // 1136 is a multiple of 16 but not of 32
    try actNvfp4(&k, s, 0, 0, 1, 1136);
}

test "expert stages pick the format's kernel and launch a block a pair by column tile" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const fun = struct {
        fn of(drv: *const cuda.Driver, i: usize) cuda.Function {
            return .{ .d = drv, .handle = @ptrFromInt(i) };
        }
    };
    const k: Kernels = .{ .mod = .{ .d = &d, .handle = null }, .fill = fun.of(&d, 1), .actq_nvfp4 = fun.of(&d, 2), .actq_mxfp8 = fun.of(&d, 3), .actq_nvfp4_codes = fun.of(&d, 4), .actq_mxfp8_codes = fun.of(&d, 5), .ex_gu_nvfp4 = fun.of(&d, 6), .ex_gu_mxfp8 = fun.of(&d, 7), .ex_down_nvfp4 = fun.of(&d, 8), .ex_down_mxfp8 = fun.of(&d, 9), .situ = fun.of(&d, 10), .ex_sum = fun.of(&d, 11), .msg = fun.of(&d, 12), .ctl_down = fun.of(&d, 13), .ctl_blend = fun.of(&d, 14), .ablit = fun.of(&d, 15) };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const g: ExpertGemm = .{ .w = 1, .scale = 2 };
    try expertGateUp(&k, s, .nvfp4, 0, 0, g, 0, 4, 2304, 2304);
    try std.testing.expectEqual(@as(usize, 6), Fake.which);
    try std.testing.expectEqual(@as(u32, 32), Fake.grid_x);
    try expertGateUp(&k, s, .mxfp8, 0, 0, g, 0, 1, 2304, 2304);
    try std.testing.expectEqual(@as(usize, 7), Fake.which);
    try expertDown(&k, s, .nvfp4, 0, 0, g, 0, 1, 2304, 1152);
    try std.testing.expectEqual(@as(usize, 8), Fake.which);
    try expertDown(&k, s, .mxfp8, 0, 0, g, 0, 1, 2304, 1152);
    try std.testing.expectEqual(@as(usize, 9), Fake.which);
    try situ(&k, s, 0, 0, 0, 1, 1152, 4.0, 25.0);
    try std.testing.expectEqual(@as(usize, 10), Fake.which);
    try std.testing.expectEqual(@as(u32, 8 * 1152 / elementwise_threads), Fake.grid_x);
    try expertSum(&k, s, 0, 0, 1, 2304);
    try std.testing.expectEqual(@as(usize, 11), Fake.which);
    const launched = Fake.count;
    try std.testing.expectError(error.BadWidth, expertGateUp(&k, s, .nvfp4, 0, 0, g, 0, 1, 2304, 1136));
    try std.testing.expectEqual(launched, Fake.count);
}

test "the loader type-checks in a build with kernels" {
    _ = &Kernels.load;
    _ = &Kernels.unload;
}
