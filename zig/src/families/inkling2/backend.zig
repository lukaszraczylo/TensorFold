//! Inkling 2 Mini on CUDA behind lanes.Backend: a slot a stream, a chunk a prompt step, one forward a token.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const decode = @import("decode.zig");
const engine = @import("engine.zig");
const head = @import("head.zig");
const ops = @import("ops.zig");
const pre = @import("prefill.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes: the engine's chunk grid, from position 0.
pub const prompt_rows: u32 = pre.default_chunk_rows;
/// Rows a verify window holds: the engine steps one token at a time.
pub const window_rows: u32 = 1;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: u32,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    rule: ?cuda.sample.Rule = null, // set when the stream samples
};

/// The slot's pick, as the device reported it: a row without a finite top value is an error.
fn checkPick(id: u32, val: f32, vocab: u32) !u32 {
    if (id >= vocab or !std.math.isFinite(val)) return error.NonFiniteLogits;
    return id;
}

fn admitWindow(w: be.Window) !void {
    if (w.rows() != window_rows) return error.WindowTooWide;
    if (w.parents != null or w.masks.len != 0) return error.StructuresUnsupported;
}

/// A kept path is the window's one row.
fn admitPath(path: []const u32) !void {
    if (path.len == 0) return error.EmptyPath;
    if (path.len != 1 or path[0] != 0) return error.WindowTooWide;
}

fn admitDraft(requests: []const be.DraftRequest) !void {
    for (requests) |r| if (r.depth != 0 or r.lanes != null) return error.NoDraftHead;
}

pub const Inkling = struct {
    gpa: std.mem.Allocator,
    c: *engine.Ctx,
    s: cuda.Stream,
    context: usize,
    lanes: std.AutoHashMapUnmanaged(*const Stream, Lane) = .empty,
    free: []bool, // slot i is free
    heads: []head.Scratch, // slot i's last row of fp32 logits and its greedy pick
    sample_image: cuda.Module,
    draws: cuda.sample.Draws,
    bf: cuda.DeviceBuffer, // the drawn row rounded to bf16, which the keyed draw reads
    rule: cuda.DeviceBuffer, // the draw's sample.Rule
    meta: cuda.DeviceBuffer, // a draw's position minus one
    out: cuda.DeviceBuffer, // the drawn token (u32)
    ring: [ring]u32 = undefined,
    next: u64 = 0,

    /// `streams` slots over `c`, whose caches hold `streams` streams of `context` positions.
    pub fn init(gpa: std.mem.Allocator, c: *engine.Ctx, s: cuda.Stream, streams: u32, context: usize) !Inkling {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const d = c.d;
        const vocab: usize = c.text.unpadded_vocab;
        const free = try gpa.alloc(bool, streams);
        errdefer gpa.free(free);
        @memset(free, true);
        const heads = try gpa.alloc(head.Scratch, streams);
        errdefer gpa.free(heads);
        var made: usize = 0;
        errdefer for (heads[0..made]) |*h| h.deinit();
        for (heads) |*h| {
            h.* = try head.Scratch.init(d, 1, vocab);
            made += 1;
        }
        var image = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer image.unload();
        const draws = try cuda.sample.Draws.resolve(image);
        var bf = try cuda.DeviceBuffer.alloc(d, vocab * 2);
        errdefer bf.free();
        var rule = try cuda.DeviceBuffer.alloc(d, @sizeOf(cuda.sample.Rule));
        errdefer rule.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        const out = try cuda.DeviceBuffer.alloc(d, 4);
        return .{ .gpa = gpa, .c = c, .s = s, .context = context, .free = free, .heads = heads, .sample_image = image, .draws = draws, .bf = bf, .rule = rule, .meta = meta, .out = out };
    }

    pub fn deinit(x: *Inkling) void {
        x.lanes.deinit(x.gpa);
        x.out.free();
        x.meta.free();
        x.rule.free();
        x.bf.free();
        x.sample_image.unload();
        for (x.heads) |*h| h.deinit();
        x.gpa.free(x.heads);
        x.gpa.free(x.free);
    }

    pub fn backend(x: *Inkling) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .reports_fill = true };

    /// The round loop's facts: one-token windows, no draft head, a stream a slot.
    pub fn facts(x: *const Inkling) lanes.Model {
        return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .max_streams = @intCast(x.free.len) };
    }

    fn self(ptr: *anyopaque) *Inkling {
        return @ptrCast(@alignCast(ptr));
    }

    fn freeSlot(x: *const Inkling) ?usize {
        return std.mem.indexOfScalar(bool, x.free, true);
    }

    fn fitsContext(x: *const Inkling, prompt: usize, max_new: usize) bool {
        return prompt + max_new + 1 <= x.context;
    }

    /// Whether a request could start now: a free slot, and its prompt and reply inside the context.
    pub fn admits(x: *const Inkling, prompt: usize, max_new: usize) bool {
        return x.fitsContext(prompt, max_new) and x.freeSlot() != null;
    }

    /// A prompt pass begins: a slot, the stream's rule, cleared conv states.
    fn opening(x: *Inkling, s: *Stream) !*Lane {
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (s.images.len > 0) return error.InklingTextOnly;
        if (!x.fitsContext(p.len, s.max_new)) return error.PastContext;
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (!got.found_existing) {
            const slot = x.freeSlot() orelse {
                x.lanes.removeByPtr(got.key_ptr);
                return error.NoFreeSlot;
            };
            x.free[slot] = false;
            got.value_ptr.* = .{ .slot = @intCast(slot) };
        }
        const l = got.value_ptr;
        l.rule = null;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            l.rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
        };
        l.len = 0;
        s.cached = 0;
        try x.c.cache.clear(x.s, l.slot);
        l.fill = 0;
        s.fill_at.store(0, .monotonic);
        return l;
    }

    /// One chunk from `l.fill` to the next grid line or the prompt's end; true once the last row's logits are written.
    fn chunk(x: *Inkling, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        const end = @min(p.len, (at / prompt_rows + 1) * prompt_rows);
        const done = end == p.len;
        try pre.chunk(x.c, x.s, l.slot, p[at..end], at, if (done) x.heads[l.slot].r.logits else null, .{});
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const x = self(ptr);
        const going = if (x.lanes.getPtr(s)) |l| l.fill != null else false;
        const l = if (going) x.lanes.getPtr(s).? else try x.opening(s);
        errdefer x.dropLane(s);
        return x.chunk(s, l);
    }

    /// The greedy pick of the slot's logits; a row without a finite top value is an error.
    fn pick(x: *Inkling, h: *const head.Scratch) !u32 {
        const r = h.r;
        try head.argmax(x.c, x.s, r.logits, 1, r.val, r.id);
        try x.s.synchronize();
        var id: u32 = undefined;
        var val: f32 = undefined;
        try cuda.memory.download(x.c.d, r.id, std.mem.asBytes(&id));
        try cuda.memory.download(x.c.d, r.val, std.mem.asBytes(&val));
        return checkPick(id, val, x.c.text.unpadded_vocab);
    }

    /// The token drawn from the slot's logits at `position`: the keyed draw over their bf16 rounding when sampling.
    fn draw(x: *Inkling, l: *const Lane, position: u64) !u32 {
        const h = &x.heads[l.slot];
        const rule = l.rule orelse return x.pick(h);
        const n: usize = x.c.text.unpadded_vocab;
        try ops.tobf(&x.c.ops, x.s, h.r.logits, 1, n, n, x.bf.ptr, n, 0, n);
        const meta = [4]i32{ @intCast(position - 1), 0, 0, 0 };
        try x.rule.uploadAsync(0, std.mem.asBytes(&rule), x.s.handle);
        try x.meta.uploadAsync(0, std.mem.asBytes(&meta), x.s.handle);
        try x.draws.draw(x.s, x.bf.ptr, n, x.rule.ptr, x.meta.ptr, 0, x.out.ptr, 1, null, null);
        try x.s.synchronize();
        var tok: u32 = undefined;
        try x.out.download(0, std.mem.asBytes(&tok));
        return tok;
    }

    fn record(x: *Inkling, tok: u32) u64 {
        const h = x.next;
        x.ring[@intCast(h % ring)] = tok;
        x.next += 1;
        return h;
    }

    /// `token` at the stream's next position through every visit, then the draw of the one after at `position`.
    fn step(x: *Inkling, l: *Lane, token: u32, position: u64) !u32 {
        if (position != l.len + 1) return error.PositionMismatch;
        try decode.step(x.c, x.s, l.slot, token, l.len, x.c.sched.n - 1, .{});
        l.len += 1;
        try head.logits(x.c, x.s, x.c.flow.r.inp, 1, x.heads[l.slot].r.logits);
        return x.draw(l, position);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        return x.record(try x.draw(l, position));
    }

    fn value(x: *Inkling, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        const t = try x.value(feed);
        return x.record(try x.step(l, t, position));
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        return x.ring[@intCast(h % ring)];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| try admitWindow(w);
        const x = self(ptr);
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (w.positions.len != window_rows) return error.PositionMismatch;
            o.sampled[0] = try x.step(l, w.pending, w.positions[0]);
        }
    }

    /// A one-row window keeps its row: the caches already hold it.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            try admitPath(path);
            if (!x.lanes.contains(w.stream)) return error.UnknownStream;
        }
    }

    /// No draft head: absorbing is a no-op, a request for drafts is refused by name.
    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        try admitDraft(requests);
    }

    fn dropLane(x: *Inkling, s: *const Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| x.free[entry.value.slot] = true;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropLane(s);
    }
};

const ts = std.testing;

/// A window of `rows` rows (the pending one and `rows - 1` host drafts), optionally with a tree's parents or masks.
fn window(rows: usize, parents: ?[]const i32, masks: []const u32) be.Window {
    const drafts = [_]u32{ 1, 2, 3 };
    return .{ .stream = undefined, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = parents, .positions = &.{}, .masks = masks };
}

test "window of one row" {
    const mask = [_]u32{1};
    const cases = [_]struct { w: be.Window, err: anyerror }{
        .{ .w = window(2, null, &.{}), .err = error.WindowTooWide },
        .{ .w = window(4, &[_]i32{ -1, 0, 1, 2 }, &.{}), .err = error.WindowTooWide },
        .{ .w = window(1, &[_]i32{-1}, &.{}), .err = error.StructuresUnsupported },
        .{ .w = window(1, null, &mask), .err = error.StructuresUnsupported },
    };
    for (cases) |k| {
        var out: [1]be.Verified = .{.{ .sampled = &.{}, .drafts = &.{} }};
        try ts.expectError(k.err, Inkling.verify(@ptrFromInt(@alignOf(Inkling)), &.{k.w}, &out));
    }
}

test "keep takes the path of the one row" {
    const x: *anyopaque = @ptrFromInt(@alignOf(Inkling));
    const w = window(1, null, &.{});
    try ts.expectError(error.WindowTooWide, Inkling.keep(x, &.{w}, &.{&.{ 0, 1 }}));
    try ts.expectError(error.WindowTooWide, Inkling.keep(x, &.{w}, &.{&.{1}}));
    try ts.expectError(error.EmptyPath, Inkling.keep(x, &.{w}, &.{&.{}}));
}

test "draft depth 0 only" {
    const x: *anyopaque = @ptrFromInt(@alignOf(Inkling));
    const base: be.DraftRequest = .{ .stream = undefined, .follow = &.{}, .rows = null, .start = 0, .position = 1, .depth = 0 };
    var deep = base;
    deep.depth = 1;
    try Inkling.draft(x, &.{base});
    try ts.expectError(error.NoDraftHead, Inkling.draft(x, &.{ base, deep }));
}

test "a pick needs a finite top value inside the vocabulary" {
    try ts.expectEqual(@as(u32, 7), try checkPick(7, 3.5, 200058));
    try ts.expectEqual(@as(u32, 0), try checkPick(0, -30, 200058));
    try ts.expectError(error.NonFiniteLogits, checkPick(200058, 1, 200058));
    try ts.expectError(error.NonFiniteLogits, checkPick(5, -std.math.inf(f32), 200058));
    try ts.expectError(error.NonFiniteLogits, checkPick(5, std.math.nan(f32), 200058));
}

test "a request needs a free slot and its prompt and reply inside the context" {
    var x: Inkling = undefined;
    var free = [_]bool{ true, true };
    x.free = &free;
    x.context = 4096;
    try ts.expect(x.admits(100, 100));
    try ts.expect(x.admits(4094, 1));
    try ts.expect(!x.admits(4095, 1));
    try ts.expect(!x.admits(4090, 100));
    free[0] = false;
    try ts.expect(x.admits(100, 100));
    free[1] = false;
    try ts.expect(!x.admits(100, 100));
    try ts.expectEqual(@as(u32, 2), x.facts().max_streams);
}

test "handles name the last ring's tokens" {
    var x: Inkling = undefined;
    x.next = 0;
    const a = x.record(11);
    const b = x.record(22);
    try ts.expectEqual(@as(u32, 11), try Inkling.read(&x, a));
    try ts.expectEqual(@as(u32, 22), try Inkling.read(&x, b));
    try ts.expectError(error.NoSuchToken, Inkling.read(&x, b + 1));
    for (0..ring) |i| _ = x.record(@intCast(i));
    try ts.expectError(error.NoSuchToken, Inkling.read(&x, a));
    try ts.expectEqual(@as(u32, ring - 1), try Inkling.read(&x, x.next - 1));
}
