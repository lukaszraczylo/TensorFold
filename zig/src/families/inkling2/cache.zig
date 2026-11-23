//! Per-visit KV caches and conv states for each stream: one device block of each, laid out stream-major.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const kernels = @import("kernels.zig");
const visits = @import("visits.zig");

/// The four conv sites of a visit: the K and V convs of attention and the two sublayer convs.
pub const Conv = enum { k, v, attn, mlp };
pub const conv_sites = std.enums.values(Conv).len;

/// tf_ink2_conv keeps a state as fp32 [conv_taps - 1, C]; its values are bf16 inputs, so the reference's bf16 state is exact.
pub const state_bytes_per_value = 4;
pub const state_rows = config.conv_taps - 1;

pub const kv8_env = "TF_INK2_KV8";

pub const Options = struct {
    context: usize,
    streams: u32,
    chunk_rows: u32,
    kv8: bool = false, // global layers' K and V caches are int8 codes with fp32 scales (TF_INK2_KV8)
};

const Slot = struct {
    layer: kernels.Layer,
    kv_heads: u32,
    kv_off: u64, // K at kv_off, V right after it
    conv_off: [conv_sites]u64,
};

pub fn convWidth(t: *const config.Text, layer: u32, which: Conv) usize {
    const h = t.heads(layer);
    return switch (which) {
        .k, .v => @as(usize, h.kv) * h.dim,
        .attn, .mlp => t.hidden,
    };
}

/// The device layout: sizes and each visit's offsets inside one stream's block.
pub const Layout = struct {
    slots: [visits.max_visits]Slot,
    n: u32,
    streams: u32,
    kv_stream: u64, // bytes of one stream's K and V for every visit
    conv_stream: u64,

    pub fn build(sched: *const visits.Schedule, t: *const config.Text, o: Options) Layout {
        var l: Layout = .{ .slots = undefined, .n = sched.n, .streams = o.streams, .kv_stream = 0, .conv_stream = 0 };
        for (sched.visits[0..sched.n], 0..) |v, i| {
            const layer = if (v.global) kernels.Layer.globalOf(@intCast(o.context), o.kv8) else kernels.Layer.sliding(t.window, o.chunk_rows);
            const kv = t.heads(v.layer).kv;
            const s = &l.slots[i];
            s.* = .{ .layer = layer, .kv_heads = kv, .kv_off = l.kv_stream, .conv_off = undefined };
            l.kv_stream += 2 * layer.cacheBytes(kv);
            for (&s.conv_off, 0..) |*off, w| {
                off.* = l.conv_stream;
                l.conv_stream += state_rows * convWidth(t, v.layer, @fromBackingInt(@intCast(w))) * state_bytes_per_value;
            }
        }
        return l;
    }

    pub fn kvTotal(l: Layout) u64 {
        return l.kv_stream * l.streams;
    }

    pub fn convTotal(l: Layout) u64 {
        return l.conv_stream * l.streams;
    }
};

/// The device bytes the budget accounts for (`budget.plan`'s kv + conv): the conv states at their bf16 reference width.
pub fn bytes(sched: *const visits.Schedule, t: *const config.Text, o: Options) u64 {
    const streams: u64 = o.streams;
    return streams * (visits.kvBytes(sched, t, o.context, o.chunk_rows, o.kv8) + sched.n * visits.convBytes(t));
}

/// Zeroes a device span on a fresh non-blocking stream and waits: a legacy-stream memset does not order against the engine's streams.
pub fn zeroSync(d: *const cuda.Driver, ptr: u64, len: usize) !void {
    if (len == 0) return;
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    const view: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
    try view.fill8(0, s.handle);
    try s.synchronize();
}

const row_bytes = kernels.head_dim * 2;

/// One key head's ring bytes: rows `first_pos..first_pos + keep` of `src` (bf16 [keep, kv_heads, head_dim]) at slot
/// pos % ring, every other slot zero. `dst` holds `ring` rows.
pub fn headRing(dst: []u8, ring: usize, kv_heads: usize, head: usize, first_pos: usize, keep: usize, src: []const u8) void {
    std.debug.assert(dst.len == ring * row_bytes and src.len == keep * kv_heads * row_bytes and keep <= ring);
    @memset(dst, 0);
    for (0..keep) |i| {
        const slot = (first_pos + i) % ring;
        @memcpy(dst[slot * row_bytes ..][0..row_bytes], src[(i * kv_heads + head) * row_bytes ..][0..row_bytes]);
    }
}

pub const Cache = struct {
    d: *const cuda.Driver,
    layout: Layout,
    kv: cuda.DeviceBuffer,
    conv: cuda.DeviceBuffer,

    /// The conv states start at zero, as `ref.model.Model._visit_state`; KV rows are read only after they are written.
    pub fn init(d: *const cuda.Driver, sched: *const visits.Schedule, t: *const config.Text, o: Options) !Cache {
        if (o.streams == 0 or o.chunk_rows == 0 or o.context == 0) return error.BadCacheOptions;
        const layout = Layout.build(sched, t, o);
        var kv = try cuda.DeviceBuffer.alloc(d, layout.kvTotal());
        errdefer kv.free();
        var conv = try cuda.DeviceBuffer.alloc(d, layout.convTotal());
        errdefer conv.free();
        try zeroSync(d, conv.ptr, conv.len);
        return .{ .d = d, .layout = layout, .kv = kv, .conv = conv };
    }

    pub fn deinit(c: *Cache) void {
        c.conv.free();
        c.kv.free();
        c.* = undefined;
    }

    fn slot(c: *const Cache, v: u32, stream: u32) *const Slot {
        std.debug.assert(v < c.layout.n and stream < c.layout.streams);
        return &c.layout.slots[v];
    }

    pub fn layerAt(c: *const Cache, v: u32) kernels.Layer {
        return c.slot(v, 0).layer;
    }

    pub fn kvAt(c: *const Cache, v: u32, stream: u32) kernels.Cache {
        const s = c.slot(v, stream);
        const k = c.kv.ptr + @as(u64, stream) * c.layout.kv_stream + s.kv_off;
        return .{ .k = k, .v = k + s.layer.cacheBytes(s.kv_heads) };
    }

    /// The address of one conv site's fp32 [3, C] state.
    pub fn convAt(c: *const Cache, v: u32, stream: u32, which: Conv) u64 {
        return c.conv.ptr + @as(u64, stream) * c.layout.conv_stream + c.slot(v, stream).conv_off[@backingInt(which)];
    }

    /// Zeroes the stream's conv states for a new sequence.
    pub fn clear(c: *Cache, s: cuda.Stream, stream: u32) !void {
        if (stream >= c.layout.streams) return error.BadStream;
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = c.conv.ptr + @as(u64, stream) * c.layout.conv_stream, .len = c.layout.conv_stream, .borrowed = true };
        try view.fill8(0, s.handle);
    }
};

const ts = std.testing;
const config_test = @import("config_test.zig");
const budget = @import("budget.zig");
const plan = @import("plan.zig");
const names = @import("names.zig");
const names_test = @import("names_test.zig");

const fake_base: u64 = 1 << 32;

pub const Mini = struct {
    cfg: config.Config,
    sched: visits.Schedule,
    prec: [config.max_layers]plan.Precision,

    pub fn init(a: std.mem.Allocator) !Mini {
        var why: config.Why = .{};
        var m: Mini = undefined;
        m.cfg = try config.parse(a, config_test.mini, &why);
        const entries = try names_test.unfold(a, @embedFile("fixtures/mini_model_header.txt"), "inkling2-mini-model");
        try names.precisions(entries, m.cfg.text.layers, &m.prec);
        const p = try plan.build(&m.cfg.text, m.prec[0..m.cfg.text.layers]);
        m.sched = try visits.build(&m.cfg, &p);
        return m;
    }
};

/// A cache over fake device addresses: the accessors never touch the driver.
pub fn fake(m: *const Mini, o: Options) Cache {
    const layout = Layout.build(&m.sched, &m.cfg.text, o);
    return .{
        .d = undefined,
        .layout = layout,
        .kv = .{ .d = undefined, .ptr = fake_base, .len = layout.kvTotal() },
        .conv = .{ .d = undefined, .ptr = fake_base + layout.kvTotal(), .len = layout.convTotal() },
    };
}

test "cache bytes" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const o: Options = .{ .context = 2048, .streams = 1, .chunk_rows = 1 };
    const l = Layout.build(&m.sched, &m.cfg.text, o);
    try ts.expectEqual(@as(u64, 8 * 2048 * 4096 + 40 * 1024 * 4096), l.kv_stream);
    try ts.expectEqual(l.kv_stream + 48 * 39936, bytes(&m.sched, &m.cfg.text, o));
}

test "cache bytes equal the budget's kv and conv" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const o: Options = .{ .context = 65536, .streams = 2, .chunk_rows = 512 };
    const b = budget.plan(&m.cfg, &m.sched, m.prec[0..m.cfg.text.layers], .{ .context = o.context, .streams = o.streams, .chunk_rows = o.chunk_rows, .embed_host = false });
    try ts.expectEqual(b.kv + b.conv, bytes(&m.sched, &m.cfg.text, o));
    try ts.expectEqual(b.kv, Layout.build(&m.sched, &m.cfg.text, o).kvTotal());
}

test "conv states are fp32 and stay inside the scratch slack of the budget" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const o: Options = .{ .context = 65536, .streams = 2, .chunk_rows = 512 };
    const l = Layout.build(&m.sched, &m.cfg.text, o);
    try ts.expectEqual(@as(u64, 2 * 48 * 39936 * 2), l.convTotal());
    try ts.expect(l.convTotal() - 2 * 48 * 39936 < budget.scratch_fixed_bytes);
}

test "stream strides" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const c = fake(&m, .{ .context = 4096, .streams = 2, .chunk_rows = 64 });
    const stride = c.kvAt(0, 1).k - c.kvAt(0, 0).k;
    try ts.expectEqual(c.layout.kv_stream, stride);
    const conv_stride = c.convAt(0, 1, .k) - c.convAt(0, 0, .k);
    for (0..m.sched.n) |v| {
        const i: u32 = @intCast(v);
        try ts.expectEqual(stride, c.kvAt(i, 1).k - c.kvAt(i, 0).k);
        try ts.expectEqual(stride, c.kvAt(i, 1).v - c.kvAt(i, 0).v);
        try ts.expectEqual(conv_stride, c.convAt(i, 1, .mlp) - c.convAt(i, 0, .mlp));
    }
}

const Range = struct { lo: u64, hi: u64 };

fn lessRange(_: void, a: Range, b: Range) bool {
    return a.lo < b.lo;
}

fn expectDisjoint(ranges: []Range, lo: u64, hi: u64) !void {
    std.mem.sort(Range, ranges, {}, lessRange);
    var at = lo;
    for (ranges) |r| {
        try ts.expect(r.lo >= at);
        try ts.expect(r.hi > r.lo);
        at = r.hi;
    }
    try ts.expect(at <= hi);
}

test "no two regions overlap and all lie inside their block" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const c = fake(&m, .{ .context = 4096, .streams = 2, .chunk_rows = 64 });
    var kv: [2 * visits.max_visits * 2]Range = undefined;
    var conv: [2 * visits.max_visits * conv_sites]Range = undefined;
    var nk: usize = 0;
    var nc: usize = 0;
    for (0..c.layout.streams) |s| {
        for (0..m.sched.n) |v| {
            const i: u32 = @intCast(v);
            const st: u32 = @intCast(s);
            const layer = c.layerAt(i);
            const bytes_kv = layer.cacheBytes(m.cfg.text.heads(m.sched.visits[v].layer).kv);
            const a = c.kvAt(i, st);
            kv[nk] = .{ .lo = a.k, .hi = a.k + bytes_kv };
            kv[nk + 1] = .{ .lo = a.v, .hi = a.v + bytes_kv };
            nk += 2;
            for (std.enums.values(Conv)) |w| {
                const width = convWidth(&m.cfg.text, m.sched.visits[v].layer, w);
                const at = c.convAt(i, st, w);
                conv[nc] = .{ .lo = at, .hi = at + state_rows * width * state_bytes_per_value };
                nc += 1;
            }
        }
    }
    try expectDisjoint(kv[0..nk], c.kv.ptr, c.kv.ptr + c.kv.len);
    try expectDisjoint(conv[0..nc], c.conv.ptr, c.conv.ptr + c.conv.len);
    try ts.expectEqual(c.kv.ptr + c.kv.len, c.kv.ptr + c.layout.kvTotal());
}

test "a visit's conv states differ from its neighbours" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const c = fake(&m, .{ .context = 2048, .streams = 1, .chunk_rows = 1 });
    try ts.expect(c.convAt(7, 0, .k) != c.convAt(6, 0, .k));
    try ts.expect(c.convAt(7, 0, .k) != c.convAt(7, 0, .v));
    try ts.expect(c.kvAt(7, 0).k != c.kvAt(6, 0).k);
    try ts.expectEqual(@as(u64, 3 * 1024 * 4), c.convAt(7, 0, .v) - c.convAt(7, 0, .k));
    try ts.expectEqual(@as(u64, 3 * 2304 * 4), c.convAt(7, 0, .mlp) - c.convAt(7, 0, .attn));
}

test "a global visit's ring holds the context and a sliding one the window plus its chunk" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const c = fake(&m, .{ .context = 2048, .streams = 1, .chunk_rows = 64 });
    try ts.expectEqual(@as(u32, 2048), c.layerAt(5).ring);
    try ts.expect(c.layerAt(5).tau);
    try ts.expectEqual(@as(u32, 1088), c.layerAt(0).ring);
    try ts.expectEqual(@as(u32, 1024), c.layerAt(0).window);
}

test "init refuses empty options before it allocates" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    var d: cuda.Driver = undefined;
    try ts.expectError(error.BadCacheOptions, Cache.init(&d, &m.sched, &m.cfg.text, .{ .context = 2048, .streams = 0, .chunk_rows = 1 }));
    try ts.expectError(error.BadCacheOptions, Cache.init(&d, &m.sched, &m.cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 0 }));
}

test "headRing places a head's rows at pos % ring and zeroes the rest" {
    const ring = 4;
    const kv_heads = 2;
    var src: [3 * kv_heads * row_bytes]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(1 + (i / row_bytes) % 200);
    var dst: [ring * row_bytes]u8 = undefined;
    headRing(&dst, ring, kv_heads, 1, 3, 3, &src);
    // positions 3, 4, 5 land in slots 3, 0, 1; the (row, head) block index is i * kv_heads + head
    try ts.expectEqual(@as(u8, 1 + 1), dst[3 * row_bytes]);
    try ts.expectEqual(@as(u8, 1 + 3), dst[0 * row_bytes]);
    try ts.expectEqual(@as(u8, 1 + 5), dst[1 * row_bytes]);
    try ts.expectEqual(@as(u8, 0), dst[2 * row_bytes]);
    try ts.expectEqual(@as(u8, 0), dst[2 * row_bytes + row_bytes - 1]);
}

test "8-bit global caches: sizes, scale offsets and 16-byte aligned regions" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const o: Options = .{ .context = 4096, .streams = 2, .chunk_rows = 64, .kv8 = true };
    const c = fake(&m, o);
    const global = c.layerAt(5);
    try ts.expect(global.q8);
    try ts.expect(!c.layerAt(0).q8);
    try ts.expectEqual(@as(usize, 4096 * 8 * 132), global.cacheBytes(8));
    try ts.expectEqual(@as(usize, 4096 * 8 * 128), global.scalesAt(8));
    try ts.expectEqual(@as(u64, global.cacheBytes(8)), c.kvAt(5, 0).v - c.kvAt(5, 0).k);
    for (0..m.sched.n) |v| {
        const a = c.kvAt(@intCast(v), 1);
        try ts.expectEqual(@as(u64, 0), a.k % 16);
        try ts.expectEqual(@as(u64, 0), a.v % 16);
    }
    try ts.expectEqual(c.layout.kv_stream * 2, c.layout.kvTotal());
    try ts.expect(c.layout.kv_stream < Layout.build(&m.sched, &m.cfg.text, .{ .context = 4096, .streams = 1, .chunk_rows = 64 }).kv_stream);
}

test "8-bit cache bytes equal the budget's kv and conv" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const o: Options = .{ .context = 65536, .streams = 2, .chunk_rows = 512, .kv8 = true };
    const b = budget.plan(&m.cfg, &m.sched, m.prec[0..m.cfg.text.layers], .{ .context = o.context, .streams = o.streams, .chunk_rows = o.chunk_rows, .embed_host = false, .kv8 = true });
    try ts.expectEqual(b.kv + b.conv, bytes(&m.sched, &m.cfg.text, o));
    try ts.expectEqual(Layout.build(&m.sched, &m.cfg.text, o).kvTotal(), b.kv);
}

test "flags off keep the bf16 layout" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try Mini.init(arena.allocator());
    const l = Layout.build(&m.sched, &m.cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 1 });
    try ts.expectEqual(@as(u64, 8 * 2048 * 4096 + 40 * 1024 * 4096), l.kv_stream);
    const g = fake(&m, .{ .context = 2048, .streams = 1, .chunk_rows = 1 }).layerAt(5);
    try ts.expect(!g.q8);
}
