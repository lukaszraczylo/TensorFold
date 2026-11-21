//! One attention sublayer visit of Inkling 2 Mini (ref.model.Model._attention, SPEC 6): projections, K/V short convs,
//! clips and norms, cache write, relative-bias attention with sink and gate, and the output projection.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const cache_mod = @import("cache.zig");
const ablit = @import("ablit.zig");
const config = @import("config.zig");
const engine = @import("engine.zig");
const kernels = @import("kernels.zig");
const loop_kernels = @import("loop_kernels.zig");
const ops = @import("ops.zig");
const plan = @import("plan.zig");
const visits = @import("visits.zig");
const weights = @import("weights.zig");

const max_tiles = kernels.tileCount(engine.max_chunk_rows, 1 << 5) + 1;

const Dev = weights.Dev;

/// A bf16 weight of `shape` at least; the loader's tensors carry their own.
fn shaped(d: Dev, comptime what: []const u8, shape: []const u64) !void {
    if (d.ptr == 0 or d.rank != shape.len or !std.mem.eql(u64, d.shape[0..d.rank], shape)) {
        if (!builtin.is_test) std.debug.print("attention weight {s}: unexpected shape\n", .{what});
        return error.WeightShape;
    }
}

/// What a call needs of its arguments before any launch.
fn validate(g: engine.Geometry, layer: kernels.Layer, v: visits.Visit, lw: *const weights.LayerW, rows: usize, p0: u64, stream: u32, streams: u32) !void {
    if (rows > g.max_rows) return error.RowsExceedScratch;
    if (stream >= streams) return error.BadStream;
    if (p0 + rows > std.math.maxInt(i32)) return error.PositionRange;
    if (layer.window == 0 and p0 + rows > layer.ring) return error.ContextExceeded;
    if (v.global and (lw.k_sconv_proj == null or lw.v_sconv_proj == null)) return error.MissingDynamicTaps;
}

/// Per-call row metadata: the positions, the one segment and its tiles.
const Meta = struct {
    pos: [engine.max_chunk_rows]i32,
    segs: [1]kernels.Seg,
    tiles: [max_tiles]kernels.Tile,
    n_tiles: usize,

    fn build(m: *Meta, rows: usize, p0: u64, group: u32) !void {
        for (m.pos[0..rows], 0..) |*p, i| p.* = @intCast(p0 + i);
        m.segs = .{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 }};
        m.n_tiles = try kernels.tiles(&m.segs, group, &m.tiles);
    }
};

fn upload(c: *const engine.Ctx, s: cuda.Stream, addr: u64, items: anytype) !void {
    const bytes = std.mem.sliceAsBytes(items);
    const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = addr, .len = bytes.len, .borrowed = true };
    try view.uploadAsync(0, bytes, s.handle);
}

/// fp32 [rows, n] = x [rows, k] times w [n, k] transposed, rounded once to bf16 into `out`, as ops.linear.
fn linear(c: *const engine.Ctx, s: cuda.Stream, x: u64, w: Dev, rows: usize, n: usize, k: usize, out: u64) !void {
    try shaped(w, "linear", &.{ n, k });
    const tmp = c.scratch.r.tmp;
    if (w.q8) {
        try ops.q8ToBf16(&c.ops, s, w.ptr, n, k, 0, n, c.dq.ptr);
        try c.dense.rows(s, x, k, c.dq.ptr, tmp, n, rows, n, k);
    } else try c.dense.rows(s, x, k, w.ptr, tmp, n, rows, n, k);
    try ops.tobf(&c.ops, s, tmp, rows, n, n, out, n, 0, n);
}

/// `ops.dense1Q8` for one row, `ops.dense1Q8Rows` for a window's.
fn denseQ8(c: *const engine.Ctx, s: cuda.Stream, segs: []const ops.DenseSeg, kdim: usize, rows: usize) !void {
    if (rows == 1) return ops.dense1Q8(&c.ops, s, segs, kdim, false, 0);
    return ops.dense1Q8Rows(&c.ops, s, segs, kdim, false, 0, rows);
}

/// Whether a call of `rows` rows takes the decode-order projections: one row, or a window's rows on 8-bit weights.
fn decodeOrder(c: *const engine.Ctx, rows: usize, w: Dev) bool {
    return rows == 1 or (c.window and w.q8 and rows <= ops.window_max_rows);
}

/// The six projections of a visit (wq, wr, gate_attn, sink on x0; wk on x1; wv on x2) in one launch, each rounded to bf16. One row, or
/// a window's rows on 8-bit weights (`Ctx.window`).
fn linearsOneRow(c: *const engine.Ctx, s: cuda.Stream, x: [3]u64, lw: *const weights.LayerW, dims: [4]usize, hidden: usize, rows: usize) !void {
    const r = c.scratch.r;
    const q_dim, const rel_dim, const heads, const kv_dim = dims;
    const parts = [_]struct { x: u64, w: Dev, out: u64, n: usize }{
        .{ .x = x[0], .w = lw.wq_du, .out = r.q_raw, .n = q_dim },
        .{ .x = x[0], .w = lw.wr_du, .out = r.r_raw, .n = rel_dim },
        .{ .x = x[0], .w = lw.gate_attn, .out = r.gate, .n = q_dim },
        .{ .x = x[0], .w = lw.attn_sink_proj, .out = r.sink, .n = heads },
        .{ .x = x[1], .w = lw.wk_dv, .out = r.k_raw, .n = kv_dim },
        .{ .x = x[2], .w = lw.wv_dv, .out = r.v_raw, .n = kv_dim },
    };
    var segs: [parts.len]ops.DenseSeg = undefined;
    for (parts, &segs) |p, *sg| {
        try shaped(p.w, "linear", &.{ p.n, hidden });
        sg.* = .{ .x = p.x, .w = p.w.ptr, .out = p.out, .n = @intCast(p.n), .block0 = 0 };
    }
    if (lw.wq_du.q8) return denseQ8(c, s, &segs, hidden, rows);
    try ops.dense1(&c.ops, s, &segs, hidden);
}

/// The K or V conv of the visit: static taps on a local layer, taps from the stream's own input on a global one.
fn shortConv(c: *const engine.Ctx, s: cuda.Stream, v: visits.Visit, which: cache_mod.Conv, w: Dev, proj: ?Dev, x_in: u64, x_raw: u64, out: u64, rows: usize, stream: u32) !void {
    const g = c.scratch.geometry;
    const kv = g.kvDim();
    var a: ops.ConvArgs = .{
        .x = x_raw,
        .state = c.cache.convAt(v.conv, stream, which),
        .w = w.ptr,
        .out = out,
        .c = @intCast(kv),
        .rows = @intCast(rows),
        .mode = 0,
        .x_bf16 = 1,
    };
    if (v.global) {
        try shaped(w, "static_w", &.{ config.conv_taps, kv });
        const taps = g.kv_heads * config.conv_taps;
        const p = proj.?;
        try shaped(p, "weight_proj", &.{ taps, g.hidden });
        const dyn = c.scratch.r.dyn;
        try c.dense.rows(s, x_in, g.hidden, p.ptr, dyn, taps, rows, taps, g.hidden);
        try loop_kernels.roundBf16(&c.loop, s, dyn, rows * taps);
        a.dyn = dyn;
        a.heads = @intCast(g.kv_heads);
    } else {
        try shaped(w, "sconv", &.{ kv, 1, config.conv_taps });
    }
    try ops.conv(&c.ops, s, a, 1);
}

/// `x` are the bf16 [rows, hidden] inputs xq, xk, xv (one pointer three times on a local layer without messages);
/// `out` receives bf16 [rows, hidden], the wo_ud projection before any edit. Rows sit at positions p0.. of `stream`.
pub fn run(c: *engine.Ctx, s: cuda.Stream, v: visits.Visit, lw: *const weights.LayerW, x: [3]u64, rows: usize, p0: u64, stream: u32, out: u64) !void {
    if (rows == 0) return;
    const g = c.scratch.geometry;
    const r = c.scratch.r;
    const t = &c.text;
    const layer = c.cache.layerAt(v.kv);
    try validate(g, layer, v, lw, rows, p0, stream, c.cache.layout.streams);
    const q_dim, const kv_dim, const rel_dim = .{ g.qDim(), g.kvDim(), g.relDim() };

    if (decodeOrder(c, rows, lw.wq_du)) {
        try linearsOneRow(c, s, x, lw, .{ q_dim, rel_dim, g.heads, kv_dim }, g.hidden, rows);
    } else {
        try linear(c, s, x[0], lw.wq_du, rows, q_dim, g.hidden, r.q_raw);
        try linear(c, s, x[0], lw.wr_du, rows, rel_dim, g.hidden, r.r_raw);
        try linear(c, s, x[0], lw.gate_attn, rows, q_dim, g.hidden, r.gate);
        try linear(c, s, x[0], lw.attn_sink_proj, rows, g.heads, g.hidden, r.sink);
        try linear(c, s, x[1], lw.wk_dv, rows, kv_dim, g.hidden, r.k_raw);
        try linear(c, s, x[2], lw.wv_dv, rows, kv_dim, g.hidden, r.v_raw);
    }
    try shortConv(c, s, v, .k, lw.k_sconv, lw.k_sconv_proj, x[1], r.k_raw, r.k_conv, rows, stream);
    try shortConv(c, s, v, .v, lw.v_sconv, lw.v_sconv_proj, x[2], r.v_raw, r.v_conv, rows, stream);

    var meta: Meta = undefined;
    try meta.build(rows, p0, g.group());
    try upload(c, s, r.pos, meta.pos[0..rows]);
    try upload(c, s, r.segs, &meta.segs);
    try upload(c, s, r.tiles, meta.tiles[0..meta.n_tiles]);
    const call: kernels.Rows = .{
        .rows = @intCast(rows),
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .pos = r.pos,
        .seg_of = r.seg_of,
        .segs = r.segs,
        .tiles = r.tiles,
        .n_tiles = @intCast(meta.n_tiles),
        .last_pos = @intCast(p0 + rows - 1),
    };
    try shaped(lw.q_l2_clip, "q_l2_clip", &.{ g.heads, g.dim });
    try shaped(lw.rel_l2_clip, "rel_l2_clip", &.{ g.heads, kernels.d_rel });
    try shaped(lw.rel_logits_proj, "rel_logits_proj", &.{ kernels.d_rel, kernels.extent });
    const kv = c.cache.kvAt(v.kv, stream);
    const prep: kernels.Prep = .{ .qk_clip = t.qk_clip, .rel_clip = t.rel_clip, .floor = @floatFromInt(t.log_floor), .alpha = t.log_alpha, .eps = t.eps };
    try kernels.prep(&c.attn, s, call, layer, prep, .{ r.q_raw, r.k_conv, r.v_conv, r.r_raw }, .{ lw.q_l2_clip.ptr, lw.rel_l2_clip.ptr }, r.q, r.rel, kv);
    try kernels.relTable(&c.attn, s, call, r.rel, lw.rel_logits_proj.ptr, r.table);
    try kernels.attend(&c.attn, s, call, layer, g.stride, g.slots(), r.q, kv, r.table, r.part_o, r.part_ml, r.sink, r.gate, r.out);
    if (decodeOrder(c, rows, lw.wo_ud)) {
        try shaped(lw.wo_ud, "linear", &.{ g.hidden, q_dim });
        const seg: ops.DenseSeg = .{ .x = r.out, .w = lw.wo_ud.ptr, .out = out, .n = @intCast(g.hidden), .block0 = 0 };
        if (lw.wo_ud.q8) try denseQ8(c, s, &.{seg}, q_dim, rows) else try ops.dense1(&c.ops, s, &.{seg}, q_dim);
    } else {
        try linear(c, s, r.out, lw.wo_ud, rows, g.hidden, q_dim, out);
    }
    try applyEdit(c, s, v.layer, out, rows);
}

/// The residual edit of every visit's output, passes and prompt rows alike, when the engine holds one.
fn applyEdit(c: *const engine.Ctx, s: cuda.Stream, layer: u32, out: u64, rows: usize) !void {
    if (c.ablit) |*st| try c.ablit_hook(&c.loop, st, s, layer, out, rows, c.scratch.geometry.hidden);
}

const ts = std.testing;
const config_test = @import("config_test.zig");

fn miniGeometry() !engine.Geometry {
    var why: config.Why = .{};
    const cfg = try config.parse(ts.allocator, config_test.mini, &why);
    return engine.Geometry.of(&cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 64 });
}

const local_visit: visits.Visit = .{ .layer = 0, .pass = 0, .kv = 0, .conv = 0, .global = false, .first_of_pass = false, .restart = false };
const global_visit: visits.Visit = .{ .layer = 5, .pass = 0, .kv = 5, .conv = 5, .global = true, .first_of_pass = false, .restart = false };

test "validate refuses what the scratch, the streams and the context cannot hold" {
    const g = try miniGeometry();
    const sliding = kernels.Layer.sliding(1024, 64);
    const global = kernels.Layer.global(2048);
    var lw: weights.LayerW = .{};
    try validate(g, sliding, local_visit, &lw, 64, 5000, 0, 2);
    try ts.expectError(error.RowsExceedScratch, validate(g, sliding, local_visit, &lw, 65, 0, 0, 1));
    try ts.expectError(error.BadStream, validate(g, sliding, local_visit, &lw, 1, 0, 1, 1));
    try ts.expectError(error.MissingDynamicTaps, validate(g, global, global_visit, &lw, 1, 0, 0, 1));
    lw.k_sconv_proj = .{ .ptr = 1, .bytes = 0, .dtype = .bf16, .shape = @splat(0), .rank = 0 };
    lw.v_sconv_proj = lw.k_sconv_proj;
    try validate(g, global, global_visit, &lw, 1, 2047, 0, 1);
    try ts.expectError(error.ContextExceeded, validate(g, global, global_visit, &lw, 2, 2047, 0, 1));
    try ts.expectError(error.PositionRange, validate(g, sliding, local_visit, &lw, 1, std.math.maxInt(i32), 0, 1));
}

test "row metadata: positions, one segment, tiles of group heads" {
    var m: Meta = undefined;
    try m.build(64, 1000, 3);
    try ts.expectEqual(@as(i32, 1000), m.pos[0]);
    try ts.expectEqual(@as(i32, 1063), m.pos[63]);
    try ts.expectEqual(kernels.Seg{ .row0 = 0, .n = 64, .p0 = 1000, .off = 0 }, m.segs[0]);
    try ts.expectEqual(@as(usize, 3), m.n_tiles); // 64 rows * 3 heads = 192 pairs, 64 a tile
    try ts.expectEqual(kernels.Tile{ .seg = 0, .m0 = 128 }, m.tiles[2]);
    try m.build(1, 7, 3);
    try ts.expectEqual(@as(usize, 1), m.n_tiles);
}

test "weight shapes are checked" {
    const good: Dev = .{ .ptr = 256, .bytes = 0, .dtype = .bf16, .shape = .{ 3072, 2304, 0, 0 }, .rank = 2 };
    try shaped(good, "x", &.{ 3072, 2304 });
    try ts.expectError(error.WeightShape, shaped(good, "x", &.{ 2304, 3072 }));
    try ts.expectError(error.WeightShape, shaped(good, "x", &.{ 3072, 2304, 1 }));
    var missing = good;
    missing.ptr = 0;
    try ts.expectError(error.WeightShape, shaped(missing, "x", &.{ 3072, 2304 }));
}

const Recorded = struct {
    var layers: [visits.max_visits]u32 = undefined;
    var rows: [visits.max_visits]usize = undefined;
    var n: usize = 0;

    fn hook(_: *const loop_kernels.Kernels, _: *const ablit.State, _: cuda.Stream, layer: u32, _: u64, rows_: usize, _: u32) anyerror!void {
        layers[n] = layer;
        rows[n] = rows_;
        n += 1;
    }
};

test "attention output edit is applied once per visit" {
    var why: config.Why = .{};
    const cfg = try config.parse(ts.allocator, config_test.mini, &why);
    const prec: [config.max_layers]plan.Precision = @splat(.nvfp4);
    const p = try plan.build(&cfg.text, prec[0..cfg.text.layers]);
    const sched = try visits.build(&cfg, &p);
    var c: engine.Ctx = undefined;
    c.scratch.geometry.hidden = 2304;
    c.ablit = @as(ablit.State, undefined);
    c.ablit_hook = Recorded.hook;
    var d: cuda.Driver = undefined;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    Recorded.n = 0;
    for (sched.visits[0..sched.n]) |v| try applyEdit(&c, s, v.layer, 0, 7);
    try ts.expectEqual(@as(usize, 48), Recorded.n);
    for (sched.visits[0..sched.n], Recorded.layers[0..Recorded.n], Recorded.rows[0..Recorded.n]) |v, l, r| {
        try ts.expectEqual(v.layer, l);
        try ts.expectEqual(@as(usize, 7), r);
    }
    c.ablit = null;
    for (sched.visits[0..sched.n]) |v| try applyEdit(&c, s, v.layer, 0, 7);
    try ts.expectEqual(@as(usize, 48), Recorded.n);
}
