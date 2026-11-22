//! TF_INK2_PF_FP4: the NVFP4 routed experts of a prompt chunk on the block-scaled FP4 mma (prefill4.cu), reading the
//! checkpoint's weights in place. The sums differ from the scalar grouped kernels in order, and the weights are not
//! rounded to bf16 first; the router, the SwiGLU and the combine are unchanged.
const std = @import("std");
const cuda = @import("cuda");
const loop_kernels = @import("loop_kernels.zig");
const plan = @import("plan.zig");

pub const env_name = "TF_INK2_PF_FP4";
pub const ctl_env_name = "TF_INK2_PF_CTL";
pub const dense_env_name = "TF_INK2_PF_BF16";
pub const mx_env_name = "TF_INK2_PF_FP8";
/// Calls of at least this many rows run the prompt kernels (the decode kernels keep the smaller ones).
pub const prompt_from = 17;
const hidden = 2304;
const inter = 1152;
const warps = 4;
const tile_cols = warps * 4 * 8;
const tile_pairs = 32;
const nvfp4_block = 16;
const slots = loop_kernels.expert_slots;

pub const Kernels = struct {
    mod: cuda.Module,
    pack: cuda.Function,
    pack8: cuda.Function,
    gu: cuda.Function,
    gu8: cuda.Function,
    down: cuda.Function,
    down8: cuda.Function,
    dense: cuda.Function,
    dense_bf16: cuda.Function,
    ctl_pair: cuda.Function,
    ctl_z: cuda.Function,

    /// `loop` holds the activation packer; the image holds the two grouped GEMMs.
    pub fn load(d: *const cuda.Driver, loop: *const loop_kernels.Kernels) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_pf4);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .pack = try loop.mod.function("tf_ink2l_actq_nvfp4_pack"),
            .pack8 = try loop.mod.function("tf_ink2l_actq_mxfp8_pack"),
            .gu = try mod.function("tf_ink2p_gu_nvfp4"),
            .gu8 = try mod.function("tf_ink2p_gu_mxfp8"),
            .down = try mod.function("tf_ink2p_down_nvfp4"),
            .down8 = try mod.function("tf_ink2p_down_mxfp8"),
            .dense = try mod.function("tf_ink2p_dense_rows"),
            .dense_bf16 = try mod.function("tf_ink2p_dense_rows_bf16"),
            .ctl_pair = try loop.mod.function("tf_ink2l_ctl_pair"),
            .ctl_z = try loop.mod.function("tf_ink2l_ctl_z"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

/// A quantised activation. NVFP4: e2m1 codes u8 [rows, k/2], e4m3 block scales u8 [rows, k/16], row amax float32 [rows].
/// MXFP8: e4m3 codes u8 [rows, k], UE8M0 exponents u8 [rows, k/32] (amax unused).
pub const Act = struct { codes: u64, scales: u64, amax: u64 };

/// Bytes of the packed activations for `max_rows` rows: the router input's rows, then the SwiGLU output's rows.
pub fn actBytes(max_rows: usize) struct { x: [3]usize, y: [3]usize } {
    const pairs = max_rows * slots;
    return .{
        .x = .{ max_rows * hidden, max_rows * hidden / nvfp4_block, max_rows * 4 },
        .y = .{ pairs * inter, pairs * inter / nvfp4_block, pairs * 4 },
    };
}

pub const Bufs = struct {
    x: [3]cuda.DeviceBuffer,
    y: [3]cuda.DeviceBuffer,
    pair: cuda.DeviceBuffer, // bf16 [max_rows, 2 * hidden]: a controller's normed (dest, msg) rows
    forget: cuda.DeviceBuffer, // fp32 [max_rows, hidden]: a controller's forget pre-activation

    pub fn init(d: *const cuda.Driver, max_rows: usize) !Bufs {
        const n = actBytes(max_rows);
        var b: Bufs = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| (if (i < 3) &b.x[i] else &b.y[i - 3]).free();
        for (n.x, &b.x) |len, *slot| {
            slot.* = try cuda.DeviceBuffer.alloc(d, len);
            made += 1;
        }
        for (n.y, &b.y) |len, *slot| {
            slot.* = try cuda.DeviceBuffer.alloc(d, len);
            made += 1;
        }
        b.pair = try cuda.DeviceBuffer.alloc(d, max_rows * 2 * hidden * 2);
        errdefer b.pair.free();
        b.forget = try cuda.DeviceBuffer.alloc(d, max_rows * hidden * 4);
        return b;
    }

    pub fn deinit(b: *Bufs) void {
        for (&b.x) |*v| v.free();
        for (&b.y) |*v| v.free();
        b.pair.free();
        b.forget.free();
    }

    pub fn actX(b: *const Bufs) Act {
        return .{ .codes = b.x[0].ptr, .scales = b.x[1].ptr, .amax = b.x[2].ptr };
    }

    pub fn actY(b: *const Bufs) Act {
        return .{ .codes = b.y[0].ptr, .scales = b.y[1].ptr, .amax = b.y[2].ptr };
    }
};

/// The state a Ctx holds when a flag is on: `experts` is TF_INK2_PF_FP4, `experts8` TF_INK2_PF_FP8, `dense` TF_INK2_PF_BF16, `ctl` TF_INK2_PF_CTL.
pub const State = struct {
    k: Kernels,
    bufs: Bufs,
    experts: bool,
    experts8: bool,
    dense: bool,
    ctl: bool,

    /// Whether a layer of precision `prec` takes the mma expert path.
    pub fn takes(st: *const State, prec: plan.Precision) bool {
        return if (prec == .nvfp4) st.experts else st.experts8;
    }
};

const dense_tile_rows = 64;
const dense_tile_cols = 128;

/// out float32 [m, n] = x bf16 [m, k] (row stride x_stride elements) times w bf16 [n, k] transposed, on the bf16 mma.
fn denseWith(f: cuda.Function, k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, kdim: usize) !void {
    if (kdim % 16 != 0 or n % 8 != 0 or x % 4 != 0 or w % 4 != 0 or x_stride % 2 != 0) return error.Misaligned;
    if (m == 0 or n == 0) return;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(@as(i64, @intCast(x_stride)));
    args.add(w);
    args.add(out);
    args.add(@as(i64, @intCast(out_stride)));
    for ([_]usize{ m, n, kdim }) |v| args.add(@as(i32, @intCast(v)));
    const grid: cuda.Dim3 = .{ .x = @intCast((n + dense_tile_cols - 1) / dense_tile_cols), .y = @intCast((m + dense_tile_rows - 1) / dense_tile_rows) };
    _ = k;
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = warps * 32 } }, s, &args);
}

pub fn denseRows(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, kdim: usize) !void {
    return denseWith(k.dense, k, s, x, x_stride, w, out, out_stride, m, n, kdim);
}

/// As denseRows with `out` bf16, rounded once.
pub fn denseRowsBf16(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, kdim: usize) !void {
    return denseWith(k.dense_bf16, k, s, x, x_stride, w, out, out_stride, m, n, kdim);
}

/// One stream of a controller on the bf16 mma, as loop_kernels.controller: `h` bf16 [rows, gate_rank] and `z` float32 [rows, hidden] are scratch.
pub fn controller(st: *const State, lk: *const loop_kernels.Kernels, s: cuda.Stream, a: loop_kernels.ControllerArgs) !void {
    if (a.rows == 0) return;
    const rank = loop_kernels.gate_rank;
    var pa: cuda.Args = .{};
    for ([_]u64{ a.dest, a.msg, st.bufs.pair.ptr }) |v| pa.add(v);
    for ([_]usize{ a.rows, hidden }) |v| pa.add(@as(i32, @intCast(v)));
    try cuda.launch.launch(st.k.ctl_pair, .{ .grid = .{ .x = @intCast(a.rows) }, .block = .{ .x = loop_kernels.actq_threads } }, s, &pa);
    try denseRowsBf16(&st.k, s, st.bufs.pair.ptr, 2 * hidden, a.down, a.h, rank, a.rows, rank, 2 * hidden);
    try denseRows(&st.k, s, a.h, rank, a.gain, a.z, hidden, a.rows, hidden, rank);
    try denseRows(&st.k, s, a.h, rank, a.forget, st.bufs.forget.ptr, hidden, a.rows, hidden, rank);
    const n = a.rows * hidden;
    var za: cuda.Args = .{};
    for ([_]u64{ a.dest, a.msg, st.bufs.forget.ptr, a.z }) |v| za.add(v);
    za.add(@as(i64, @intCast(n)));
    const z_threads = 256;
    try cuda.launch.launch(st.k.ctl_z, .{ .grid = .{ .x = @intCast((n + z_threads - 1) / z_threads) }, .block = .{ .x = z_threads } }, s, &za);
    try loop_kernels.blendApply(lk, s, a.z, a.norm, a.out, a.rows);
}

/// Quantises bf16 `x` [rows, kdim] as the layer's precision does (tf_ink2l_actq_*), into `a`; `out` (float32 [rows, kdim], or 0) takes the dequantised values too.
pub fn pack(k: *const Kernels, prec: plan.Precision, s: cuda.Stream, x: u64, out: u64, a: Act, rows: usize, kdim: usize) !void {
    const block: usize = if (prec == .nvfp4) nvfp4_block else loop_kernels.mxfp8_block;
    if (kdim == 0 or kdim % block != 0) return error.BadWidth;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(out);
    if (prec == .nvfp4) {
        for ([_]u64{ a.amax, a.codes, a.scales }) |v| args.add(v);
    } else {
        for ([_]u64{ a.codes, a.scales }) |v| args.add(v);
    }
    for ([_]usize{ rows, kdim }) |v| args.add(@as(i32, @intCast(v)));
    try cuda.launch.launch(if (prec == .nvfp4) k.pack else k.pack8, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = loop_kernels.actq_threads } }, s, &args);
}

fn grouped(f: cuda.Function, prec: plan.Precision, s: cuda.Stream, a: Act, act_div: usize, offsets: u64, items: u64, g: loop_kernels.ExpertGemm, out: u64, rows: usize, experts: usize, n: usize, kdim: usize) !void {
    if (n == 0 or n % tile_cols != 0 or kdim == 0 or kdim % 128 != 0) return error.BadWidth;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    if (prec == .nvfp4) {
        for ([_]u64{ a.codes, a.scales, a.amax }) |v| args.add(v);
    } else {
        for ([_]u64{ a.codes, a.scales }) |v| args.add(v);
    }
    args.add(@as(i32, @intCast(act_div)));
    for ([_]u64{ offsets, items, g.w, g.scale }) |v| args.add(v);
    if (prec == .nvfp4) for ([_]u64{ g.amax, g.outer }) |v| args.add(v);
    args.add(out);
    for ([_]usize{ n, kdim }) |v| args.add(@as(i32, @intCast(v)));
    const pair_tiles = (rows + tile_pairs - 1) / tile_pairs;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n / tile_cols), .y = @intCast(experts), .z = @intCast(pair_tiles) }, .block = .{ .x = warps * 32 } }, s, &args);
}

/// As loop_kernels.expertGateUpGrouped: `a` holds the rows' activations; pair p lands at gu row p.
pub fn gateUp(k: *const Kernels, prec: plan.Precision, s: cuda.Stream, a: Act, offsets: u64, items: u64, g: loop_kernels.ExpertGemm, gu: u64, rows: usize, experts: usize, n: usize, kdim: usize) !void {
    return grouped(if (prec == .nvfp4) k.gu else k.gu8, prec, s, a, slots, offsets, items, g, gu, rows, experts, n, kdim);
}

/// As loop_kernels.expertDownGrouped: `a` holds one row a pair.
pub fn down(k: *const Kernels, prec: plan.Precision, s: cuda.Stream, a: Act, offsets: u64, items: u64, g: loop_kernels.ExpertGemm, dn: u64, rows: usize, experts: usize, n: usize, kdim: usize) !void {
    return grouped(if (prec == .nvfp4) k.down else k.down8, prec, s, a, 1, offsets, items, g, dn, rows, experts, n, kdim);
}

test "packed activation sizes" {
    const n = actBytes(512);
    try std.testing.expectEqual(@as(usize, 512 * 2304), n.x[0]);
    try std.testing.expectEqual(@as(usize, 512 * 144), n.x[1]);
    try std.testing.expectEqual(@as(usize, 4096 * 1152), n.y[0]);
    try std.testing.expectEqual(@as(usize, 4096 * 72), n.y[1]);
    try std.testing.expectEqual(@as(usize, 4096 * 4), n.y[2]);
}

test "a width the tiles do not cover is refused before any launch" {
    var d: cuda.Driver = undefined;
    const f: cuda.Function = .{ .d = &d, .handle = null };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const a: Act = .{ .codes = 0, .scales = 0, .amax = 0 };
    const g: loop_kernels.ExpertGemm = .{ .w = 0, .scale = 0 };
    try std.testing.expectError(error.BadWidth, grouped(f, .nvfp4, s, a, 8, 0, 0, g, 0, 512, 128, 100, 2304));
    try std.testing.expectError(error.BadWidth, grouped(f, .mxfp8, s, a, 8, 0, 0, g, 0, 512, 128, 2304, 100));
    try std.testing.expectError(error.BadWidth, pack(&.{ .mod = undefined, .pack = f, .pack8 = f, .gu = f, .gu8 = f, .down = f, .down8 = f, .dense = f, .dense_bf16 = f, .ctl_pair = f, .ctl_z = f }, .nvfp4, s, 0, 0, a, 4, 100));
}

test "dense rows refuses a width or address the fragments cannot load" {
    var d: cuda.Driver = undefined;
    const f: cuda.Function = .{ .d = &d, .handle = null };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const k: Kernels = .{ .mod = undefined, .pack = f, .pack8 = f, .gu = f, .gu8 = f, .down = f, .down8 = f, .dense = f, .dense_bf16 = f, .ctl_pair = f, .ctl_z = f };
    try std.testing.expectError(error.Misaligned, denseRows(&k, s, 0, 2304, 0, 0, 3072, 512, 3072, 100));
    try std.testing.expectError(error.Misaligned, denseRows(&k, s, 0, 2304, 0, 0, 20, 512, 20, 2304));
    try denseRows(&k, s, 0, 2304, 0, 0, 3072, 0, 3072, 2304);
}

test "each flag selects its own precision" {
    var st: State = undefined;
    st.experts = true;
    st.experts8 = false;
    try std.testing.expect(st.takes(.nvfp4));
    try std.testing.expect(!st.takes(.mxfp8));
    st.experts = false;
    st.experts8 = true;
    try std.testing.expect(!st.takes(.nvfp4));
    try std.testing.expect(st.takes(.mxfp8));
}
