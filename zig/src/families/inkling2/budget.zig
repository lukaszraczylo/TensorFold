//! Device memory plan for the looped engine: resident weights, per-visit KV and conv state, chunk scratch.
const std = @import("std");
const st = @import("safetensors");
const config = @import("config.zig");
const names = @import("names.zig");
const layer_plan = @import("plan.zig");
const visits = @import("visits.zig");

const MiB: u64 = 1 << 20;

pub const LayerRange = struct { first: u32, last: u32 };

pub const Options = struct {
    context: usize,
    streams: u32,
    chunk_rows: u32,
    embed_host: bool,
    layers: ?LayerRange = null, // stages a partial upload for dev gates
    kv8: bool = false, // global layers' K and V caches are 8-bit (TF_INK2_KV8)
};

pub const Budget = struct { weights: u64, kv: u64, conv: u64, scratch: u64, total: u64 };

/// Hidden-wide bytes of chunk scratch per row, fitted to `tf-ink2-check prefill` at 128 and 512 rows (123.7 and 300.6 MB).
const scratch_row_hidden_bytes: u64 = 200;

pub fn scratchRowBytes(c: *const config.Text) u64 {
    return scratch_row_hidden_bytes * c.hidden;
}

/// Attention partials and routing buffers, independent of the chunk; the fit stays at or above both measured points.
pub const scratch_fixed_bytes: u64 = 64 * MiB;

const layer_prefix = "model.llm.layers.";
const vision_prefix = "model.visual.";
const embed_name = "model.llm.embed.weight";

fn layerOf(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, layer_prefix)) return null;
    const rest = name[layer_prefix.len..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..end], 10) catch null;
}

const Sum = struct {
    o: Options,
    bytes: u64 = 0,

    pub fn put(s: *Sum, name: []const u8, dtype: st.DType, dims: []const u64) !void {
        if (std.mem.startsWith(u8, name, vision_prefix)) return;
        if (s.o.embed_host and std.mem.eql(u8, name, embed_name)) return;
        if (s.o.layers) |r| {
            if (layerOf(name)) |l| if (l < r.first or l > r.last) return;
        }
        var n: u64 = dtype.size();
        for (dims) |d| n *= d;
        s.bytes += n;
    }
};

/// Text tensors held on the device; `layers` limits per-layer tensors, global ones count once.
pub fn weightBytes(c: *const config.Config, prec: []const layer_plan.Precision, o: Options) u64 {
    var sum: Sum = .{ .o = o };
    // the sink never fails, and the map fails only through it
    names.all(c, prec, .model, &sum) catch unreachable;
    return sum.bytes;
}

pub fn plan(c: *const config.Config, s: *const visits.Schedule, prec: []const layer_plan.Precision, o: Options) Budget {
    const streams: u64 = o.streams;
    const weights = weightBytes(c, prec, o);
    const kv = streams * visits.kvBytes(s, &c.text, o.context, o.chunk_rows, o.kv8);
    const conv = streams * s.n * visits.convBytes(&c.text);
    const scratch = o.chunk_rows * scratchRowBytes(&c.text) + scratch_fixed_bytes;
    return .{ .weights = weights, .kv = kv, .conv = conv, .scratch = scratch, .total = weights + kv + conv + scratch };
}

pub fn check(b: Budget, limit: u64) error{OverBudget}!void {
    if (b.total > limit) return error.OverBudget;
}

const t = std.testing;
const config_test = @import("config_test.zig");
const names_test = @import("names_test.zig");
const GiB: u64 = 1 << 30;
const mini_fixture = @embedFile("fixtures/mini_model_header.txt");
const embed_bytes: u64 = 201024 * 2304 * 2;

const Mini = struct {
    cfg: config.Config,
    prec: [config.max_layers]layer_plan.Precision,
    sched: visits.Schedule,

    fn init(a: std.mem.Allocator) !Mini {
        var why: config.Why = .{};
        var m: Mini = undefined;
        m.cfg = try config.parse(a, config_test.mini, &why);
        const entries = try names_test.unfold(a, mini_fixture, "inkling2-mini-model");
        try names.precisions(entries, m.cfg.text.layers, &m.prec);
        const p = try layer_plan.build(&m.cfg.text, m.prec[0..m.cfg.text.layers]);
        m.sched = try visits.build(&m.cfg, &p);
        return m;
    }

    fn precs(m: *const Mini) []const layer_plan.Precision {
        return m.prec[0..m.cfg.text.layers];
    }

    fn budget(m: *const Mini, o: Options) Budget {
        return plan(&m.cfg, &m.sched, m.precs(), o);
    }
};

const serving: Options = .{ .context = 65536, .streams = 2, .chunk_rows = 512, .embed_host = false };
const dev_gate: Options = .{ .context = 2048, .streams = 1, .chunk_rows = 64, .embed_host = true };

test "full text weights" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const entries = try names_test.unfold(arena.allocator(), mini_fixture, "inkling2-mini-model");
    var want: u64 = 0;
    for (entries.keys(), entries.values()) |name, e| {
        if (std.mem.startsWith(u8, name, vision_prefix)) continue;
        var n: u64 = e.dtype.size();
        for (e.shape[0..e.rank]) |d| n *= d;
        want += n;
    }
    var o = serving;
    o.embed_host = false;
    const full = weightBytes(&m.cfg, m.precs(), o);
    try t.expectEqual(want, full);
    try t.expect(full >= 17 * GiB + GiB / 2 and full <= 17 * GiB + GiB * 7 / 10);
    o.embed_host = true;
    try t.expectEqual(@as(u64, 926_318_592), embed_bytes);
    try t.expectEqual(full - embed_bytes, weightBytes(&m.cfg, m.precs(), o));
}

test "kv at serving size" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    try t.expectEqual(@as(u64, 2 * 2_399_141_888), m.budget(serving).kv);
}

test "serving total fits 32 GiB" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const b = (try Mini.init(arena.allocator())).budget(serving);
    try t.expect(b.total < 28 * GiB);
    try check(b, 32 * GiB);
    try t.expectEqual(b.weights + b.kv + b.conv + b.scratch, b.total);
}

test "dev gate fits 18 GiB" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const b = (try Mini.init(arena.allocator())).budget(dev_gate);
    try t.expect(b.total < 18 * GiB);
}

test "over budget" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const b = (try Mini.init(arena.allocator())).budget(serving);
    try t.expectError(error.OverBudget, check(b, 16 * GiB));
}

test "staged layers" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    var o = dev_gate;
    o.layers = .{ .first = 0, .last = 5 };
    const w = weightBytes(&m.cfg, m.precs(), o);
    try t.expect(w >= 4 * GiB + GiB / 5 and w <= 4 * GiB + GiB * 7 / 10);
}

test "scratch formula" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    try t.expectEqual(@as(u64, 460800), scratchRowBytes(&m.cfg.text));
    try t.expectEqual(@as(u64, 64 * 460800) + scratch_fixed_bytes, m.budget(dev_gate).scratch);
    const measured = [_]struct { rows: u32, bytes: u64 }{ .{ .rows = 128, .bytes = 123_664_384 }, .{ .rows = 512, .bytes = 300_560_896 } };
    for (measured) |p| {
        const est = p.rows * scratchRowBytes(&m.cfg.text) + scratch_fixed_bytes;
        try t.expect(est >= p.bytes and est - p.bytes <= p.bytes / 10);
    }
}

test "8-bit global caches shrink the kv bytes by their codes and scales" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    var o: Options = .{ .context = 262144, .streams = 1, .chunk_rows = 512, .embed_host = true };
    const sliding_kv: u64 = 40 * 1536 * 4096;
    try t.expectEqual(8 * GiB + sliding_kv, m.budget(o).kv);
    o.kv8 = true;
    // 8 visits x 262144 tokens x 8 heads x 2 (K, V) x (128 codes + 4 scale bytes)
    const want: u64 = 8 * 262144 * 8 * 2 * 132 + sliding_kv;
    try t.expectEqual(want, m.budget(o).kv);
    try t.expectEqual(@as(u64, 4 * GiB + GiB / 8), want - sliding_kv);
    // only the kv line moves
    const off = m.budget(.{ .context = 262144, .streams = 1, .chunk_rows = 512, .embed_host = true });
    try t.expectEqual(off.weights, m.budget(o).weights);
    try t.expectEqual(off.conv, m.budget(o).conv);
    try t.expectEqual(off.scratch, m.budget(o).scratch);
}

test "8-bit global caches save 3.875 GiB of the total at a 256k window" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    var o: Options = .{ .context = 262144, .streams = 1, .chunk_rows = 512, .embed_host = true };
    const off = m.budget(o).total;
    o.kv8 = true;
    try t.expectEqual(@as(u64, 3 * GiB + GiB * 7 / 8), off - m.budget(o).total);
    try check(m.budget(o), off - 3 * GiB);
    try t.expectError(error.OverBudget, check(m.budget(.{ .context = 262144, .streams = 1, .chunk_rows = 512, .embed_host = true }), off - 3 * GiB));
}
