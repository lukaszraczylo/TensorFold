//! A stream's state at a prompt position for the core prompt cache: the global visits' keys and values below `at`, each
//! sliding visit's last window - 1 positions, the conv states, and the draft head's context.
const std = @import("std");
const cuda = @import("cuda");
const cache = @import("cache.zig");
const drafter = @import("drafter.zig");
const kernels = @import("kernels.zig");
const visits = @import("visits.zig");

const Allocator = std.mem.Allocator;
const head_dim = kernels.head_dim;
const bf16_bytes = 2;

/// A kept state: `buf` holds the target's rows, then its conv states; `draft` the head's context (TF_INK2_MTP).
pub const Saved = struct {
    at: u32,
    buf: cuda.DeviceBuffer,
    draft: ?cuda.DeviceBuffer = null,
};

/// Positions a sliding visit keeps at `at`: the next row's window reaches back window - 1 keys.
pub fn tail(window: u32, at: u32) u32 {
    return @min(at, window - 1);
}

/// One key head's K or V rows of a visit: `row` bytes a position, `ring` slots, at `base`.
fn plane(base: u64, h: usize, ring: u32, row: u64) u64 {
    return base + h * @as(u64, ring) * row;
}

/// The target's runs for `slot` at `at`, in image order: each global visit's K then V head planes (8-bit: codes, then
/// scales), then each sliding visit's, then the conv states whole. `f` gets a run's device address, image offset and length.
pub fn each(c: *const cache.Cache, sched: *const visits.Schedule, slot: u32, at: u32, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: u64, len: u64) anyerror!void) !void {
    var off: u64 = 0;
    for ([_]bool{ true, false }) |global| for (sched.visits[0..sched.n], 0..) |v, i| {
        if (v.global != global) continue;
        const layer = c.layerAt(@intCast(i));
        const kv = c.layout.slots[i].kv_heads;
        const a = c.kvAt(@intCast(i), slot);
        if (global and at > layer.ring) return error.PastContext;
        for ([_]u64{ a.k, a.v }) |base| for (0..kv) |h| {
            if (global and layer.q8) {
                try f(ctx, plane(base, h, layer.ring, head_dim), off, @as(u64, at) * head_dim);
                off += @as(u64, at) * head_dim;
                try f(ctx, plane(base, h, layer.ring, kernels.kv8_scale_bytes) + layer.scalesAt(kv), off, @as(u64, at) * kernels.kv8_scale_bytes);
                off += @as(u64, at) * kernels.kv8_scale_bytes;
                continue;
            }
            const row: u64 = head_dim * bf16_bytes;
            const first: u64 = if (global) 0 else at - tail(layer.window, at);
            var p = first;
            while (p < at) {
                const s = p % layer.ring;
                const run = @min(at - p, layer.ring - s);
                try f(ctx, plane(base, h, layer.ring, row) + s * row, off, run * row);
                off += run * row;
                p += run;
            }
        };
    };
    try f(ctx, c.conv.ptr + @as(u64, slot) * c.layout.conv_stream, off, c.layout.conv_stream);
}

/// The image's parts as `each` lays it out, for the disk store: a global plane keeps its row size (a longer state's
/// plane begins with a shorter one's), the rest is the state's own. `draft` bytes, when the head keeps context, close it.
pub fn parts(c: *const cache.Cache, sched: *const visits.Schedule, at: u32, draft: u64, out: anytype, a: Allocator) !void {
    for ([_]bool{ true, false }) |global| for (sched.visits[0..sched.n], 0..) |v, i| {
        if (v.global != global) continue;
        const layer = c.layerAt(@intCast(i));
        if (global and at > layer.ring) return error.PastContext;
        for (0..2) |_| for (0..c.layout.slots[i].kv_heads) |_| {
            if (!global) {
                try out.append(a, .{ .bytes = @as(u64, tail(layer.window, at)) * head_dim * bf16_bytes });
            } else if (layer.q8) {
                try out.append(a, .{ .bytes = @as(u64, at) * head_dim, .row = head_dim });
                try out.append(a, .{ .bytes = @as(u64, at) * kernels.kv8_scale_bytes, .row = kernels.kv8_scale_bytes });
            } else try out.append(a, .{ .bytes = @as(u64, at) * head_dim * bf16_bytes, .row = head_dim * bf16_bytes });
        };
    };
    try out.append(a, .{ .bytes = c.layout.conv_stream });
    if (draft > 0) try out.append(a, .{ .bytes = draft });
}

const Count = struct {
    n: *u64,
    fn add(c: Count, _: u64, _: u64, len: u64) anyerror!void {
        c.n.* += len;
    }
};

/// The bytes `each` copies at `at` positions.
pub fn bytes(c: *const cache.Cache, sched: *const visits.Schedule, at: u32) !u64 {
    var n: u64 = 0;
    try each(c, sched, 0, at, Count{ .n = &n }, Count.add);
    return n;
}

const Copy = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    buf: u64,
    out: bool,

    fn run(c: Copy, dev: u64, off: u64, len: u64) anyerror!void {
        const kept = c.buf + off;
        const src = if (c.out) dev else kept;
        const dst = if (c.out) kept else dev;
        try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
    }
};

fn copyAll(d: *const cuda.Driver, s: cuda.Stream, c: *const cache.Cache, sched: *const visits.Schedule, draft: ?*const drafter.Drafter, slot: u32, k: *const Saved, out: bool) !void {
    try each(c, sched, slot, k.at, Copy{ .d = d, .s = s, .buf = k.buf.ptr, .out = out }, Copy.run);
    if (draft) |x| try x.each(slot, k.at, Copy{ .d = d, .s = s, .buf = (k.draft orelse return error.NoDraftState).ptr, .out = out }, Copy.run);
}

/// Copies `slot`'s state at `at` into buffers of its own, queued on `s` behind the chunk that ends there.
pub fn save(gpa: Allocator, d: *const cuda.Driver, s: cuda.Stream, c: *const cache.Cache, sched: *const visits.Schedule, draft: ?*const drafter.Drafter, slot: u32, at: u32) !*Saved {
    if (at == 0) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .buf = try cuda.DeviceBuffer.alloc(d, try bytes(c, sched, at)) };
    errdefer k.buf.free();
    if (draft) |x| k.draft = try cuda.DeviceBuffer.alloc(d, x.snapBytes(at));
    errdefer if (k.draft) |*b| b.free();
    errdefer s.synchronize() catch {}; // queued copies finish before their buffers go
    try copyAll(d, s, c, sched, draft, slot, k, true);
    return k;
}

/// Makes `slot` hold `k`'s state, queued on `s` before the next chunk.
pub fn restore(d: *const cuda.Driver, s: cuda.Stream, c: *const cache.Cache, sched: *const visits.Schedule, draft: ?*const drafter.Drafter, slot: u32, k: *const Saved) !void {
    try copyAll(d, s, c, sched, draft, slot, k, false);
}

/// Frees a state once the stream's queued copies of it are done.
pub fn drop(gpa: Allocator, s: cuda.Stream, k: *Saved) void {
    s.synchronize() catch {};
    k.buf.free();
    if (k.draft) |*b| b.free();
    gpa.destroy(k);
}

const ts = std.testing;

const Runs = struct {
    list: *std.ArrayList([3]u64),
    fn add(r: Runs, dev: u64, off: u64, len: u64) anyerror!void {
        try r.list.append(ts.allocator, .{ dev, off, len });
    }
};

test "a state is the global planes below at, the sliding tails, then the conv states, in contiguous image order" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    const c = cache.fake(&m, .{ .context = 4096, .streams = 2, .chunk_rows = 512 });
    const at = 2000;
    var list: std.ArrayList([3]u64) = .empty;
    defer list.deinit(ts.allocator);
    try each(&c, &m.sched, 1, at, Runs{ .list = &list }, Runs.add);
    var end: u64 = 0;
    for (list.items) |r| {
        try ts.expectEqual(end, r[1]); // runs tile the image
        end += r[2];
    }
    const kv_row = 8 * head_dim * bf16_bytes;
    const global = 8 * 2 * at * kv_row;
    const sliding = 40 * 2 * (1024 - 1) * kv_row;
    try ts.expectEqual(global + sliding + c.layout.conv_stream, end);
    try ts.expectEqual(end, try bytes(&c, &m.sched, at));
    const last = list.items[list.items.len - 1];
    try ts.expectEqual(c.conv.ptr + c.layout.conv_stream, last[0]); // slot 1's block
    try ts.expectEqual(c.layout.conv_stream, last[2]);
}

test "a sliding visit's tail is window - 1 positions, wrapped at the ring, and every run stays inside its head's plane" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    const c = cache.fake(&m, .{ .context = 4096, .streams = 1, .chunk_rows = 512 });
    const ring: u64 = c.layerAt(0).ring;
    const row: u64 = head_dim * bf16_bytes;
    const at: u32 = @intCast(ring + 300); // the ring has wrapped: positions at-1023..at-1 straddle its end
    var list: std.ArrayList([3]u64) = .empty;
    defer list.deinit(ts.allocator);
    try each(&c, &m.sched, 0, at, Runs{ .list = &list }, Runs.add);
    const k0 = c.kvAt(0, 0).k;
    var found: [2][3]u64 = undefined;
    var n: usize = 0;
    for (list.items) |r| {
        if (r[0] < k0 or r[0] >= k0 + ring * row) continue; // visit 0's head 0 K plane
        try ts.expect(r[0] + r[2] <= k0 + ring * row);
        found[n] = r;
        n += 1;
    }
    try ts.expectEqual(@as(usize, 2), n); // the tail's two runs: the plane's end, then its start
    try ts.expectEqual(@as(u64, 1023 * row), found[0][2] + found[1][2]);
    try ts.expectEqual(k0 + ((at - 1023) % ring) * row, found[0][0]);
    try ts.expectEqual(k0, found[1][0]);
}

test "8-bit global planes keep codes then scales for the same positions" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    const c = cache.fake(&m, .{ .context = 4096, .streams = 1, .chunk_rows = 512, .kv8 = true });
    const at = 1000;
    var list: std.ArrayList([3]u64) = .empty;
    defer list.deinit(ts.allocator);
    try each(&c, &m.sched, 0, at, Runs{ .list = &list }, Runs.add);
    const g = c.layerAt(5);
    const a = c.kvAt(5, 0);
    var codes: ?[3]u64 = null;
    var scales: ?[3]u64 = null;
    for (list.items) |r| {
        if (r[0] == a.k) codes = r;
        if (r[0] == a.k + g.scalesAt(8)) scales = r;
    }
    try ts.expectEqual(@as(u64, at * head_dim), codes.?[2]);
    try ts.expectEqual(@as(u64, at * kernels.kv8_scale_bytes), scales.?[2]);
    try ts.expectEqual(codes.?[1] + codes.?[2], scales.?[1]);
    try ts.expect(try bytes(&c, &m.sched, at) < try bytes(&cache.fake(&m, .{ .context = 4096, .streams = 1, .chunk_rows = 512 }), &m.sched, at));
}

test "a state past a global ring is refused" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    const c = cache.fake(&m, .{ .context = 2048, .streams = 1, .chunk_rows = 512 });
    try ts.expectError(error.PastContext, bytes(&c, &m.sched, 2049));
}

const Part = struct { bytes: u64, row: u32 = 0 };

test "the disk parts tile the image: each part ends where a run does, and global planes keep their row size" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    for ([_]bool{ false, true }) |kv8| {
        const c = cache.fake(&m, .{ .context = 4096, .streams = 1, .chunk_rows = 512, .kv8 = kv8 });
        for ([_]u32{ 1, 700, 1023, 1024, 3000 }) |at| {
            var list: std.ArrayList(Part) = .empty;
            defer list.deinit(ts.allocator);
            try parts(&c, &m.sched, at, 0, &list, ts.allocator);
            var runs: std.ArrayList([3]u64) = .empty;
            defer runs.deinit(ts.allocator);
            try each(&c, &m.sched, 0, at, Runs{ .list = &runs }, Runs.add);
            var ends: std.ArrayList(u64) = .empty;
            defer ends.deinit(ts.allocator);
            for (runs.items) |r| try ends.append(ts.allocator, r[1] + r[2]);
            var off: u64 = 0;
            for (list.items) |p| {
                off += p.bytes;
                try ts.expect(std.mem.indexOfScalar(u64, ends.items, off) != null);
                if (p.row > 0) try ts.expectEqual(@as(u64, at) * p.row, p.bytes);
            }
            try ts.expectEqual(try bytes(&c, &m.sched, at), off);
        }
    }
}

test "a draft part closes the image" {
    var arena = std.heap.ArenaAllocator.init(ts.allocator);
    defer arena.deinit();
    const m = try cache.Mini.init(arena.allocator());
    const c = cache.fake(&m, .{ .context = 4096, .streams = 1, .chunk_rows = 512 });
    var list: std.ArrayList(Part) = .empty;
    defer list.deinit(ts.allocator);
    try parts(&c, &m.sched, 100, 77, &list, ts.allocator);
    try ts.expectEqual(@as(u64, 77), list.items[list.items.len - 1].bytes);
    try ts.expectEqual(c.layout.conv_stream, list.items[list.items.len - 2].bytes);
}
