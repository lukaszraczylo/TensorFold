//! Inkling 2 Mini's routed MoE for a few rows (decode): router, quantised activations, expert GEMVs, SwiGLU, combine.
const std = @import("std");
const cuda = @import("cuda");
const loop_kernels = @import("loop_kernels.zig");
const ops = @import("ops.zig");
const plan = @import("plan.zig");
const weights = @import("weights.zig");

pub const hidden = 2304;
pub const inter = 1152;
pub const experts = 128;
pub const top = loop_kernels.expert_slots;
pub const route_scale: f32 = 8.0;
pub const gate_cap: f32 = 4.0;
pub const up_cap: f32 = 25.0;

/// What a forward pass needs besides its stream: the driver and the two kernel images.
pub const Env = struct {
    driver: *const cuda.Driver,
    loop: *const loop_kernels.Kernels,
    ops: *const ops.Kernels,
};

/// Bytes of each scratch buffer for `max_rows` rows.
pub const Sizes = struct {
    scores: usize, // router sigmoid scores, float32 [rows, experts]
    pick: usize, // int32 [rows, 8]
    gamma: usize, // float32 [rows, 8]
    act: usize, // quantised input, float32 [rows, hidden]
    gu: usize, // bf16 [rows * 8, 2 * inter]
    y: usize, // bf16 [rows * 8, inter]
    yq: usize, // quantised y, float32 [rows * 8, inter]
    dn: usize, // float32 [rows * 8, hidden]

    pub fn of(max_rows: usize) Sizes {
        const pairs = max_rows * top;
        return .{
            .scores = max_rows * experts * 4,
            .pick = pairs * 4,
            .gamma = pairs * 4,
            .act = max_rows * hidden * 4,
            .gu = pairs * 2 * inter * 2,
            .y = pairs * inter * 2,
            .yq = pairs * inter * 4,
            .dn = pairs * hidden * 4,
        };
    }
};

/// Device buffers for `max_rows` rows; one forward at a time may use it.
pub const Scratch = struct {
    max_rows: usize,
    scores: cuda.DeviceBuffer,
    pick: cuda.DeviceBuffer,
    gamma: cuda.DeviceBuffer,
    act: cuda.DeviceBuffer,
    gu: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    yq: cuda.DeviceBuffer,
    dn: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver, max_rows: usize) !Scratch {
        const n = Sizes.of(max_rows);
        var scores = try cuda.DeviceBuffer.alloc(d, n.scores);
        errdefer scores.free();
        var pick = try cuda.DeviceBuffer.alloc(d, n.pick);
        errdefer pick.free();
        var gamma = try cuda.DeviceBuffer.alloc(d, n.gamma);
        errdefer gamma.free();
        var act = try cuda.DeviceBuffer.alloc(d, n.act);
        errdefer act.free();
        var gu = try cuda.DeviceBuffer.alloc(d, n.gu);
        errdefer gu.free();
        var y = try cuda.DeviceBuffer.alloc(d, n.y);
        errdefer y.free();
        var yq = try cuda.DeviceBuffer.alloc(d, n.yq);
        errdefer yq.free();
        const dn = try cuda.DeviceBuffer.alloc(d, n.dn);
        return .{ .max_rows = max_rows, .scores = scores, .pick = pick, .gamma = gamma, .act = act, .gu = gu, .y = y, .yq = yq, .dn = dn };
    }

    pub fn deinit(sc: *Scratch) void {
        for ([_]*cuda.DeviceBuffer{ &sc.scores, &sc.pick, &sc.gamma, &sc.act, &sc.gu, &sc.y, &sc.yq, &sc.dn }) |b| b.free();
        sc.* = undefined;
    }
};

fn actQuant(k: *const loop_kernels.Kernels, s: cuda.Stream, prec: plan.Precision, x: u64, out: u64, rows: usize, kdim: usize) !void {
    return switch (prec) {
        .nvfp4 => loop_kernels.actNvfp4(k, s, x, out, rows, kdim),
        .mxfp8 => loop_kernels.actMxfp8(k, s, x, out, rows, kdim),
    };
}

/// ref.ops.moe(x, layer, ckpt, cfg, act_quant=True): `x` and `out` are bf16 [rows, 2304]; the picks and weights stay in `sc`.
pub fn forward(env: *const Env, s: cuda.Stream, layer: *const weights.LayerW, prec: plan.Precision, x: u64, out: u64, rows: usize, sc: *Scratch) !void {
    if (rows == 0) return;
    if (rows > sc.max_rows) return error.RowsExceedScratch;
    const ex = layer.mlp.experts;
    const k = env.loop;
    const pairs = rows * top;

    try ops.rowdot(env.ops, s, x, layer.mlp.gate.ptr, sc.scores.ptr, rows, experts, hidden, false, true);
    try ops.topk(env.ops, s, sc.scores.ptr, layer.mlp.gate_bias.ptr, sc.pick.ptr, sc.gamma.ptr, rows, experts, top, route_scale);

    try actQuant(k, s, prec, x, sc.act.ptr, rows, hidden);
    try loop_kernels.expertGateUp(k, s, prec, sc.act.ptr, sc.pick.ptr, gemm(ex.w13, ex.w13_scale, ex.w13_amax, ex.w13_outer), sc.gu.ptr, rows, 2 * inter, hidden);
    try loop_kernels.situ(k, s, sc.gu.ptr, sc.gamma.ptr, sc.y.ptr, rows, inter, gate_cap, up_cap);
    try actQuant(k, s, prec, sc.y.ptr, sc.yq.ptr, pairs, inter);
    try loop_kernels.expertDown(k, s, prec, sc.yq.ptr, sc.pick.ptr, gemm(ex.w2, ex.w2_scale, ex.w2_amax, ex.w2_outer), sc.dn.ptr, rows, hidden, inter);
    try loop_kernels.expertSum(k, s, sc.dn.ptr, out, rows, hidden);
}

fn gemm(w: weights.Dev, scale: weights.Dev, amax: ?weights.Dev, outer: ?weights.Dev) loop_kernels.ExpertGemm {
    return .{ .w = w.ptr, .scale = scale.ptr, .amax = if (amax) |a| a.ptr else 0, .outer = if (outer) |o| o.ptr else 0 };
}

test "scratch sizes" {
    const n = Sizes.of(16);
    try std.testing.expectEqual(@as(usize, 16 * 8 * 2304 * 2), n.gu);
    try std.testing.expectEqual(@as(usize, 16 * 8 * 1152 * 2), n.y);
    try std.testing.expectEqual(@as(usize, 16 * 8 * 1152 * 4), n.yq);
    try std.testing.expectEqual(@as(usize, 16 * 8 * 2304 * 4), n.dn);
    try std.testing.expectEqual(@as(usize, 16 * 128 * 4), n.scores);
    try std.testing.expectEqual(@as(usize, 16 * 8 * 4), n.pick);
    try std.testing.expectEqual(n.pick, n.gamma);
}

test "forward refuses more rows than the scratch holds" {
    var d: cuda.Driver = undefined;
    var sc: Scratch = undefined;
    sc.max_rows = 4;
    var env: Env = .{ .driver = &d, .loop = undefined, .ops = undefined };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const layer: weights.LayerW = .{};
    try std.testing.expectError(error.RowsExceedScratch, forward(&env, s, &layer, .nvfp4, 0, 0, 5, &sc));
    try forward(&env, s, &layer, .nvfp4, 0, 0, 0, &sc);
}
