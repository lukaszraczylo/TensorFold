//! Output head of Inkling 2 Mini (ref.model.Model._head, SPEC 3): the final-normed bf16 `h` the last visit leaves in the
//! carried input times the first `unpadded_vocab` rows of `unembed.weight` into fp32 logits, the softcap in fp32, and the
//! greedy pick over those logits.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const engine = @import("engine.zig");
const ops = @import("ops.zig");
const weights = @import("weights.zig");

const bf16_bytes = 2;
const f32_bytes = 4;
const id_bytes = 4;
const region_align = 256;

/// The unembedding the head reads: bf16 [vocab, hidden] on the device with every real token's row.
fn unembedding(t: *const config.Text, w: *const weights.Weights) !u64 {
    const d = w.unembed;
    if (d.ptr == 0) return error.Inkling2MissingWeight;
    if (d.dtype != .bf16 or d.rank != 2 or d.shape[1] != t.hidden or d.shape[0] < t.unpadded_vocab) return error.Inkling2HeadShape;
    return d.ptr;
}

/// Logits of `rows` bf16 rows `h` [rows, hidden]: out_f32 [rows, unpadded_vocab] = cap * tanh(h . unembed^T / cap), the
/// product accumulated and left in fp32 (SPEC 3 steps 4-6).
pub fn logits(c: *engine.Ctx, s: cuda.Stream, h: u64, rows: usize, out_f32: u64) !void {
    const t = &c.text;
    const w = try unembedding(t, c.w);
    const n: usize = t.unpadded_vocab;
    try c.dense.rows(s, h, t.hidden, w, out_f32, n, rows, n, t.hidden);
    try ops.softcap(&c.ops, s, out_f32, rows * n, t.logit_cap);
}

/// Each of `rows` logit rows [rows, unpadded_vocab]: the first index of its largest value into `id` (i32 [rows]), the value
/// into `val` (f32 [rows]).
pub fn argmax(c: *engine.Ctx, s: cuda.Stream, logit_rows: u64, rows: usize, val: u64, id: u64) !void {
    const n: usize = c.text.unpadded_vocab;
    try ops.top1(&c.ops, s, logit_rows, rows, n, n, 0, val, id);
}

/// Device buffers of a head call: the logits and the pick. `of(0, rows, n)` sizes them.
pub const Regions = struct {
    logits: u64, // fp32 [rows, n]
    val: u64, // fp32 [rows]
    id: u64, // int32 [rows]
    total: u64,

    pub fn of(base: u64, rows: usize, n: usize) Regions {
        var at = base;
        var r: Regions = undefined;
        r.logits = take(&at, rows * n * f32_bytes);
        r.val = take(&at, rows * f32_bytes);
        r.id = take(&at, rows * id_bytes);
        r.total = at - base;
        return r;
    }

    fn take(at: *u64, len: usize) u64 {
        const p = at.*;
        at.* += std.mem.alignForward(u64, len, region_align);
        return p;
    }
};

pub const Scratch = struct {
    mem: cuda.DeviceBuffer,
    r: Regions,

    pub fn init(d: *const cuda.Driver, rows: usize, n: usize) !Scratch {
        const mem = try cuda.DeviceBuffer.alloc(d, Regions.of(0, rows, n).total);
        return .{ .mem = mem, .r = Regions.of(mem.ptr, rows, n) };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.mem.free();
        sc.* = undefined;
    }
};

/// Device bytes of the head: the unembedding rows it reads (already in the weights) and its scratch.
pub fn deviceBytes(t: *const config.Text, rows: usize) struct { unembed: u64, scratch: u64 } {
    return .{
        .unembed = @as(u64, t.vocab) * t.hidden * bf16_bytes,
        .scratch = Regions.of(0, rows, t.unpadded_vocab).total,
    };
}

/// The first index of the largest value of `row`; NaN is never picked, a row without a number gives null. The host twin of
/// `argmax` for the tests and the tool's cross-check.
pub fn hostArgmax(row: []const f32) ?u32 {
    var best: ?u32 = null;
    for (row, 0..) |x, i| {
        if (std.math.isNan(x)) continue;
        if (best == null or x > row[best.?]) best = @intCast(i);
    }
    return best;
}

const ts = std.testing;
const config_test = @import("config_test.zig");

test "the head's shape comes from the config, not from constants" {
    var why: config.Why = .{};
    const cfg = try config.parse(ts.allocator, config_test.mini, &why);
    try ts.expectEqual(@as(u32, 200058), cfg.text.unpadded_vocab);
    try ts.expectEqual(@as(u32, 201024), cfg.text.vocab);
    try ts.expectEqual(@as(f32, 30), cfg.text.logit_cap);
    const b = deviceBytes(&cfg.text, 1);
    try ts.expectEqual(@as(u64, 926_318_592), b.unembed);
    // 200058 fp32 logits and two scalars, each region rounded up to 256 bytes
    try ts.expectEqual(@as(u64, 800_256 + 256 + 256), b.scratch);
}

test "the scratch regions are aligned, ordered and disjoint" {
    const r = Regions.of(1 << 20, 4, 200058);
    try ts.expect(r.logits % region_align == 0 and r.val % region_align == 0 and r.id % region_align == 0);
    try ts.expect(r.val >= r.logits + 4 * 200058 * f32_bytes);
    try ts.expect(r.id >= r.val + 4 * f32_bytes);
    try ts.expect(r.id + 4 * id_bytes <= (1 << 20) + r.total);
}

test "the unembedding must be a bf16 table wide enough for every real token" {
    var why: config.Why = .{};
    const cfg = try config.parse(ts.allocator, config_test.mini, &why);
    var w: weights.Weights = undefined;
    w.unembed = .{ .ptr = 0, .bytes = 0, .dtype = .bf16, .shape = @splat(0), .rank = 0 };
    try ts.expectError(error.Inkling2MissingWeight, unembedding(&cfg.text, &w));
    w.unembed = .{ .ptr = 4096, .bytes = 0, .dtype = .bf16, .shape = .{ cfg.text.vocab, cfg.text.hidden, 0, 0 }, .rank = 2 };
    try ts.expectEqual(@as(u64, 4096), try unembedding(&cfg.text, &w));
    w.unembed.shape[0] = cfg.text.unpadded_vocab - 1;
    try ts.expectError(error.Inkling2HeadShape, unembedding(&cfg.text, &w));
    w.unembed.shape = .{ cfg.text.vocab, cfg.text.hidden + 8, 0, 0 };
    try ts.expectError(error.Inkling2HeadShape, unembedding(&cfg.text, &w));
    w.unembed.shape = .{ cfg.text.vocab, cfg.text.hidden, 0, 0 };
    w.unembed.dtype = .f32;
    try ts.expectError(error.Inkling2HeadShape, unembedding(&cfg.text, &w));
}

test "the host pick takes the first of equal maxima and skips NaN" {
    const nan = std.math.nan(f32);
    try ts.expectEqual(@as(?u32, 2), hostArgmax(&.{ 1, 3, 7, 7, 0 }));
    try ts.expectEqual(@as(?u32, 1), hostArgmax(&.{ nan, -std.math.inf(f32), nan }));
    try ts.expectEqual(@as(?u32, 3), hostArgmax(&.{ nan, 1, nan, 9 }));
    try ts.expectEqual(@as(?u32, null), hostArgmax(&.{ nan, nan }));
    try ts.expectEqual(@as(?u32, null), hostArgmax(&.{}));
}
