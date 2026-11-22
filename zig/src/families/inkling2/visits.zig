//! The looped model's visit schedule: each core layer runs once a pass, and each visit owns its KV cache and conv state.
const std = @import("std");
const config = @import("config.zig");
const plan = @import("plan.zig");
const kernels = @import("kernels.zig");

pub const max_visits = 64;

pub const Visit = struct {
    layer: u32,
    pass: u32,
    kv: u32, // index of the visit's KV cache
    conv: u32, // index of the visit's conv state set
    global: bool,
    first_of_pass: bool,
    restart: bool, // the first core layer of a later pass
};

pub const Schedule = struct {
    visits: [max_visits]Visit,
    n: u32,
    sliding: u32,
    global_: u32,
};

fn add(s: *Schedule, p: *const plan.Plan, layer: u32, pass: u32, loop: config.Loop) void {
    const first = layer == loop.core_start;
    const global = p.at(layer).kind == .global;
    s.visits[s.n] = .{
        .layer = layer,
        .pass = pass,
        .kv = s.n,
        .conv = s.n,
        .global = global,
        .first_of_pass = first,
        .restart = first and pass > 0,
    };
    s.n += 1;
    if (global) s.global_ += 1 else s.sliding += 1;
}

/// Prelude once, the core `passes` times, then the coda; a model with no loop has no schedule.
pub fn build(c: *const config.Config, p: *const plan.Plan) error{NotLoop}!Schedule {
    const loop = c.loop orelse return error.NotLoop;
    std.debug.assert(loop.visitCount(c.text.layers) <= max_visits);
    var s: Schedule = .{ .visits = undefined, .n = 0, .sliding = 0, .global_ = 0 };
    for (0..loop.core_start) |l| add(&s, p, @intCast(l), 0, loop);
    for (0..loop.passes) |pass| {
        for (loop.core_start..loop.core_end) |l| add(&s, p, @intCast(l), @intCast(pass), loop);
    }
    for (loop.core_end..c.text.layers) |l| add(&s, p, @intCast(l), 0, loop);
    return s;
}

/// Bytes of one visit's conv state: the last `conv_taps - 1` rows of the k, v and two hidden-wide streams, bf16.
pub fn convBytes(c: *const config.Text) usize {
    const kv_dim = @as(usize, c.local.kv) * c.local.dim;
    return (config.conv_taps - 1) * (2 * kv_dim + 2 * @as(usize, c.hidden)) * 2;
}

/// All visits' K and V caches for one stream of `rows`-row calls over `context` tokens; the global ones 8-bit when `kv8`.
pub fn kvBytes(s: *const Schedule, c: *const config.Text, context: usize, rows: u32, kv8: bool) usize {
    var total: usize = 0;
    for (s.visits[0..s.n]) |v| {
        const layer = if (v.global) kernels.Layer.globalOf(@intCast(context), kv8) else kernels.Layer.sliding(c.window, rows);
        total += layer.cacheBytes(c.heads(v.layer).kv) * 2;
    }
    return total;
}

const t = std.testing;
const config_test = @import("config_test.zig");

const Fixture = struct {
    cfg: config.Config,
    plan: plan.Plan,

    fn init(json: []const u8) !Fixture {
        var why: config.Why = .{};
        const cfg = try config.parse(t.allocator, json, &why);
        const prec: [config.max_layers]plan.Precision = @splat(.nvfp4);
        return .{ .cfg = cfg, .plan = try plan.build(&cfg.text, prec[0..cfg.text.layers]) };
    }
};

fn mini() !struct { fx: Fixture, s: Schedule } {
    const fx = try Fixture.init(config_test.mini);
    return .{ .fx = fx, .s = try build(&fx.cfg, &fx.plan) };
}

fn expectVisit(s: Schedule, i: usize, layer: u32, pass: u32) !void {
    try t.expectEqual(layer, s.visits[i].layer);
    try t.expectEqual(pass, s.visits[i].pass);
}

test "48 visits in order" {
    const m = try mini();
    try t.expectEqual(@as(u32, 48), m.s.n);
    const want = [_][3]u32{ .{ 0, 0, 0 }, .{ 5, 5, 0 }, .{ 6, 6, 0 }, .{ 17, 17, 0 }, .{ 18, 6, 1 }, .{ 29, 17, 1 }, .{ 30, 6, 2 }, .{ 41, 17, 2 }, .{ 42, 18, 0 }, .{ 47, 23, 0 } };
    for (want) |w| try expectVisit(m.s, w[0], w[1], w[2]);
}

test "kinds" {
    const m = try mini();
    try t.expectEqual(@as(u32, 40), m.s.sliding);
    try t.expectEqual(@as(u32, 8), m.s.global_);
    const want = [_]usize{ 5, 11, 17, 23, 29, 35, 41, 47 };
    var k: usize = 0;
    for (m.s.visits[0..m.s.n], 0..) |v, i| {
        if (!v.global) continue;
        try t.expectEqual(want[k], i);
        k += 1;
    }
    try t.expectEqual(want.len, k);
}

test "visit slots are unique and sized per kind" {
    const m = try mini();
    for (m.s.visits[0..m.s.n], 0..) |v, i| {
        try t.expectEqual(@as(u32, @intCast(i)), v.kv);
        try t.expectEqual(@as(u32, @intCast(i)), v.conv);
    }
    try t.expect(m.s.visits[6].kv != m.s.visits[18].kv);
    try t.expect(m.s.visits[18].kv != m.s.visits[30].kv);
    try t.expect(m.s.visits[6].kv != m.s.visits[30].kv);
}

test "restart flags" {
    const m = try mini();
    for (m.s.visits[0..m.s.n], 0..) |v, i| {
        try t.expectEqual(i == 18 or i == 30, v.restart);
        try t.expectEqual(i == 6 or i == 18 or i == 30, v.first_of_pass);
    }
}

test "sizes" {
    const m = try mini();
    try t.expectEqual(@as(usize, 39936), convBytes(&m.fx.cfg.text));
    try t.expectEqual(@as(usize, 2_399_141_888), kvBytes(&m.s, &m.fx.cfg.text, 65536, 512, false));
    try t.expectEqual(@as(usize, 8 * 65536 * 4096 + 40 * 1536 * 4096), kvBytes(&m.s, &m.fx.cfg.text, 65536, 512, false));
    // 8-bit global caches: 8 visits x 65536 tokens x 8 heads x 2 (K, V) x (128 codes + 4 scale bytes)
    try t.expectEqual(@as(usize, 8 * 65536 * 8 * 2 * 132 + 40 * 1536 * 4096), kvBytes(&m.s, &m.fx.cfg.text, 65536, 512, true));
}

test "small is refused" {
    const fx = try Fixture.init(config_test.tiny);
    try t.expectError(error.NotLoop, build(&fx.cfg, &fx.plan));
}
