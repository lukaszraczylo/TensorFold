//! TF_INK2_PF_FP4: the NVFP4 routed experts of a prompt chunk on the block-scaled FP4 mma (prefill4.cu), reading the
//! checkpoint's weights in place. The sums differ from the scalar grouped kernels in order, and the weights are not
//! rounded to bf16 first; the router, the SwiGLU and the combine are unchanged.
const std = @import("std");
const cuda = @import("cuda");
const loop_kernels = @import("loop_kernels.zig");
const plan = @import("plan.zig");

pub const env_name = "TF_INK2_PF_FP4";
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
        return b;
    }

    pub fn deinit(b: *Bufs) void {
        for (&b.x) |*v| v.free();
        for (&b.y) |*v| v.free();
    }

    pub fn actX(b: *const Bufs) Act {
        return .{ .codes = b.x[0].ptr, .scales = b.x[1].ptr, .amax = b.x[2].ptr };
    }

    pub fn actY(b: *const Bufs) Act {
        return .{ .codes = b.y[0].ptr, .scales = b.y[1].ptr, .amax = b.y[2].ptr };
    }
};

/// The state a Ctx holds when a flag is on: `experts` is TF_INK2_PF_FP4, `experts8` TF_INK2_PF_FP8, `dense` TF_INK2_PF_BF16.
pub const State = struct {
    k: Kernels,
    bufs: Bufs,
    experts: bool,
    experts8: bool,
    dense: bool,

    /// Whether a layer of precision `prec` takes the mma expert path.
    pub fn takes(st: *const State, prec: plan.Precision) bool {
        return if (prec == .nvfp4) st.experts else st.experts8;
    }
};

const dense_tile_rows = 64;
const dense_tile_cols = 128;

/// out float32 [m, n] = x bf16 [m, k] (row stride x_stride elements) times w bf16 [n, k] transposed, on the bf16 mma.
pub fn denseRows(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, kdim: usize) !void {
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
    try cuda.launch.launch(k.dense, .{ .grid = grid, .block = .{ .x = warps * 32 } }, s, &args);
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
    try std.testing.expectError(error.BadWidth, pack(&.{ .mod = undefined, .pack = f, .pack8 = f, .gu = f, .gu8 = f, .down = f, .down8 = f, .dense = f }, .nvfp4, s, 0, 0, a, 4, 100));
}

test "dense rows refuses a width or address the fragments cannot load" {
    var d: cuda.Driver = undefined;
    const f: cuda.Function = .{ .d = &d, .handle = null };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const k: Kernels = .{ .mod = undefined, .pack = f, .pack8 = f, .gu = f, .gu8 = f, .down = f, .down8 = f, .dense = f };
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
