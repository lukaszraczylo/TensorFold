//! Inkling 2 Mini on CUDA behind lanes.Backend: a slot a stream, a chunk a prompt step, one forward a token.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const decode = @import("decode.zig");
const engine = @import("engine.zig");
const head = @import("head.zig");
const prof = @import("prof.zig");
const ops = @import("ops.zig");
const pre = @import("prefill.zig");
const drafter_mod = @import("drafter.zig");
const snap = @import("snap.zig");
const platform = @import("sys");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a step takes: the engine's chunk grid, from position 0.
pub const prompt_rows: u32 = pre.default_chunk_rows;
/// Rows a verify window holds without the draft head: the engine steps one token at a time.
pub const window_rows: u32 = 1;
/// Rows a verify window holds under TF_INK2_MTP (the window kernels' most).
pub const mtp_rows: u32 = ops.window_max_rows;
/// Drafts the head holds for a stream: its block's proposals.
const most_drafts = drafter_mod.proposals;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: u32,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    rule: ?cuda.sample.Rule = null, // set when the stream samples
    win_start: u64 = 0, // the cache length the last verify window began at
    win_rows: u32 = 0, // its rows
    written: u64 = 0, // positions [0, written) are in the draft head's context
    drafts: [most_drafts]u32 = undefined, // the head's held drafts
    n_drafts: u32 = 0,
    chain: [most_drafts]f32 = undefined, // the head's probability for each held draft (TF_INK2_VCAP)
};

/// The draft head and what a window needs around it (TF_INK2_MTP).
const Mtp = struct {
    head: drafter_mod.Drafter,
    taps: cuda.DeviceBuffer, // bf16 [slots][rows][taps][hidden]: the rows the last forward of a slot tapped
    rows: u32, // the engine's chunk rows
    row_bytes: u64,
    logits: cuda.DeviceBuffer, // fp32 [mtp_rows][vocab]: a window's rows
    val: cuda.DeviceBuffer, // fp32 [mtp_rows]
    id: cuda.DeviceBuffer, // i32 [mtp_rows]

    fn slotTaps(m: *const Mtp, slot: u32) u64 {
        return m.taps.ptr + @as(u64, slot) * m.rows * m.row_bytes;
    }
};

/// Where the draft head's weights are read from.
pub const DraftSource = struct { io: std.Io, dir: []const u8 };

/// Forward ms by window width on the GB10 (rowcost, DENSE8 and HEAD8, ctx 2048): what the round loop prices a window at.
const window_costs = [_]lanes.config.Cost{
    .{ .width = 1, .ms = 29.76 },
    .{ .width = 2, .ms = 35.56 },
    .{ .width = 3, .ms = 39.19 },
    .{ .width = 4, .ms = 43.24 },
    .{ .width = 5, .ms = 46.69 },
    .{ .width = 6, .ms = 50.00 },
    .{ .width = 7, .ms = 53.89 },
    .{ .width = 8, .ms = 58.21 },
};
/// One head block (context write, 3 layers, 7 Markov steps) in ms (draftacc).
const draft_ms = 10.5;

/// The slot's pick, as the device reported it: a row without a finite top value is an error.
fn checkPick(id: u32, val: f32, vocab: u32) !u32 {
    if (id >= vocab or !std.math.isFinite(val)) return error.NonFiniteLogits;
    return id;
}

fn admitWindow(w: be.Window, max_rows: u32) !void {
    if (w.rows() > max_rows) return error.WindowTooWide;
    if (w.parents != null or w.masks.len != 0) return error.StructuresUnsupported;
}

/// A kept path is a prefix of the window's rows.
fn admitPath(path: []const u32, max_rows: u32) !void {
    if (path.len == 0) return error.EmptyPath;
    if (path.len > max_rows) return error.WindowTooWide;
    for (path, 0..) |row, i| if (row != i) return error.WindowTooWide;
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
    rows: u32 = window_rows, // a verify window's most rows
    mtp: ?Mtp = null,

    /// `streams` slots over `c`, whose caches hold `streams` streams of `context` positions; `draft` (TF_INK2_MTP) loads the draft head.
    pub fn init(gpa: std.mem.Allocator, c: *engine.Ctx, s: cuda.Stream, streams: u32, context: usize, draft_src: ?DraftSource) !Inkling {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const d = c.d;
        try prof.install(d);
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
        var out = try cuda.DeviceBuffer.alloc(d, 4);
        errdefer out.free();
        var mtp: ?Mtp = null;
        if (draft_src) |src| mtp = try openMtp(gpa, c, src, streams, context);
        return .{ .gpa = gpa, .c = c, .s = s, .context = context, .free = free, .heads = heads, .sample_image = image, .draws = draws, .bf = bf, .rule = rule, .meta = meta, .out = out, .rows = if (mtp != null) mtp_rows else window_rows, .mtp = mtp };
    }

    fn openMtp(gpa: std.mem.Allocator, c: *engine.Ctx, src: DraftSource, streams: u32, context: usize) !Mtp {
        const d = c.d;
        const m = c.w.cfg.mtp orelse return error.NoDraftHead;
        c.rec = try engine.Rec.init(d, gpa, c.cache, &c.sched, &c.text);
        errdefer if (c.rec) |*r| r.deinit();
        var head_run = try drafter_mod.Drafter.init(gpa, src.io, c, src.dir, streams, context);
        errdefer head_run.deinit();
        const row_bytes = @as(u64, m.taps_n) * c.text.hidden * 2;
        const chunk_rows = c.scratch.geometry.max_rows;
        var taps = try cuda.DeviceBuffer.alloc(d, @as(usize, streams) * chunk_rows * row_bytes);
        errdefer taps.free();
        const n: usize = c.text.unpadded_vocab;
        var logits = try cuda.DeviceBuffer.alloc(d, mtp_rows * n * 4);
        errdefer logits.free();
        var val = try cuda.DeviceBuffer.alloc(d, mtp_rows * 4);
        errdefer val.free();
        const id = try cuda.DeviceBuffer.alloc(d, mtp_rows * 4);
        return .{ .head = head_run, .taps = taps, .rows = chunk_rows, .row_bytes = row_bytes, .logits = logits, .val = val, .id = id };
    }

    pub fn deinit(x: *Inkling) void {
        if (x.mtp) |*m| {
            m.id.free();
            m.val.free();
            m.logits.free();
            m.taps.free();
            m.head.deinit();
        }
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

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .probabilities = probabilities, .release = release, .reports_fill = true };

    /// The round loop's facts: one-token windows and no draft head, or (TF_INK2_MTP) windows of up to `mtp_rows` rows with the
    /// head's block of drafts; a stream a slot.
    pub fn facts(x: *const Inkling) lanes.Model {
        if (x.mtp == null) return .{ .exact_width = window_rows, .gpu_tokens = false, .mtp = false, .speculate = false, .max_streams = @intCast(x.free.len) };
        return .{
            .exact_width = mtp_rows,
            .gpu_tokens = false,
            .mtp = true,
            .speculate = true,
            .speculate_early = false,
            .copy_drafts = false,
            .fixed_draft_block = true,
            .drafts = most_drafts,
            .draft_probabilities = x.mtp.?.head.vcap and x.mtp.?.head.vcap_floor == null,
            .draft_confidence = x.mtp.?.head.vcap_floor,
            .window_costs = &window_costs,
            .mtp_step_ms = draft_ms,
            .max_streams = @intCast(x.free.len),
        };
    }

    /// The tap rows of the next forward of `l`'s slot go to its buffer; `window` selects the window kernels and the conv records.
    fn arm(x: *Inkling, l: *const Lane, wide: bool) void {
        const m = if (x.mtp) |*m| m else return;
        x.c.taps = .{ .dst = m.slotTaps(l.slot), .taps = x.c.w.cfg.mtp.?.taps_n };
        x.c.window = wide;
        x.c.recording = wide;
    }

    /// The draft head takes `rows` tapped rows (row `first` of the slot's buffer on) as positions `pos0`.. of its context.
    fn absorb(x: *Inkling, l: *Lane, row0: usize, rows: usize, pos0: u64) !void {
        const m = if (x.mtp) |*m| m else return;
        const taps = m.slotTaps(l.slot);
        var done: usize = 0;
        while (done < rows) {
            const n = @min(drafter_mod.write_rows, rows - done);
            try m.head.writeContext(x.s, l.slot, taps + (row0 + done) * m.row_bytes, n, pos0 + done);
            done += n;
        }
        try m.head.setRnnContext(x.s, l.slot, taps, row0 + rows - 1);
        l.written = pos0 + rows;
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
        l.written = 0;
        l.n_drafts = 0;
        s.cached = 0;
        try x.c.cache.clear(x.s, l.slot);
        l.fill = x.resumeAt(s, l, p.len);
        s.fill_at.store(@intCast(l.fill.?), .monotonic);
        return l;
    }

    pub fn draftHead(x: *const Inkling) ?*const drafter_mod.Drafter {
        return if (x.mtp) |*m| &m.head else null;
    }

    /// Bytes a kept state at `at` takes (the store's charge).
    pub fn snapBytes(x: *const Inkling, at: u32) u64 {
        const target = snap.bytes(x.c.cache, &x.c.sched, at) catch return std.math.maxInt(u32);
        return target + if (x.draftHead()) |h| h.snapBytes(at) else 0;
    }

    /// The stream's state at `at`, copied behind the chunk that ends there.
    pub fn saveState(x: *Inkling, owner: ?*anyopaque, at: u32) !*snap.Saved {
        const s: *const Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if ((l.fill orelse l.len) != at or (x.mtp != null and l.written != at)) return error.NotAtMark; // a state is copied only where the pass stands
        return snap.save(x.gpa, x.c.d, x.s, x.c.cache, &x.c.sched, x.draftHead(), l.slot, at);
    }

    /// A state of `at` positions in buffers of its own, for the disk store to fill.
    pub fn blankState(x: *Inkling, at: u32) !*snap.Saved {
        const k = try x.gpa.create(snap.Saved);
        errdefer x.gpa.destroy(k);
        k.* = .{ .at = at, .buf = try cuda.DeviceBuffer.alloc(x.c.d, try snap.bytes(x.c.cache, &x.c.sched, at)) };
        errdefer k.buf.free();
        if (x.draftHead()) |h| k.draft = try cuda.DeviceBuffer.alloc(x.c.d, h.snapBytes(at));
        return k;
    }

    pub fn dropState(x: *Inkling, k: *snap.Saved) void {
        snap.drop(x.gpa, x.s, k);
    }

    /// The kept state's position, restored into the slot (0: none, or not this prompt's); any grid row resumes exactly.
    fn resumeAt(x: *Inkling, s: *Stream, l: *Lane, len: usize) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const snap.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or (x.mtp != null and k.draft == null)) {
            s.reuse_failed = true;
            return 0;
        }
        snap.restore(x.c.d, x.s, x.c.cache, &x.c.sched, x.draftHead(), l.slot, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        l.written = k.at;
        s.cached = k.at;
        return k.at;
    }

    /// One chunk from `l.fill` to the next grid line or the prompt's end; true once the last row's logits are written.
    fn chunk(x: *Inkling, s: *Stream, l: *Lane) !bool {
        const p = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        var end = @min(p.len, (at / x.c.scratch.geometry.max_rows + 1) * x.c.scratch.geometry.max_rows);
        for (s.reuse.marks) |mk| if (mk > at and mk < end) {
            end = mk;
        };
        const done = end == p.len;
        x.arm(l, false);
        prof.promptBegin(x.s.handle);
        try pre.chunk(x.c, x.s, l.slot, p[at..end], at, if (done) x.heads[l.slot].r.logits else null, .{});
        try x.absorb(l, 0, end - at, at);
        prof.promptEnd();
        s.fill_at.store(@intCast(end), .monotonic);
        l.fill = if (done) null else end;
        if (done) l.len = p.len;
        if (s.reuse.hook) |hk| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null) hk.at(hk.ptr, s, @intCast(end));
        return done;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        prof.roundEnd();
        const l = try x.opening(s);
        errdefer x.dropLane(s);
        while (!try x.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream, _: u32) anyerror!bool {
        const x = self(ptr);
        prof.roundEnd();
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
        _ = try x.pick(h); // checks the top logit only: a NaN elsewhere in the row is not seen here
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
        prof.begin(x.s.handle);
        x.arm(l, false);
        try decode.step(x.c, x.s, l.slot, token, l.len, x.c.sched.n - 1, .{});
        try x.absorb(l, 0, 1, l.len);
        l.len += 1;
        prof.at(prof.tag_head);
        try head.logits(x.c, x.s, x.c.flow.r.inp, 1, x.heads[l.slot].r.logits);
        const tok = try x.draw(l, position);
        prof.end();
        return tok;
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

    /// Each row of the window's logits drawn at its position: the greedy picks in one pass, a sampling stream's keyed draws a row at a time.
    fn drawRows(x: *Inkling, m: *const Mtp, l: *const Lane, positions: []const u64, out: []u32) !void {
        const n: usize = x.c.text.unpadded_vocab;
        const rule = l.rule orelse {
            try head.argmax(x.c, x.s, m.logits.ptr, out.len, m.val.ptr, m.id.ptr);
            try x.s.synchronize();
            var ids: [mtp_rows]u32 = undefined;
            var vals: [mtp_rows]f32 = undefined;
            try m.id.download(0, std.mem.sliceAsBytes(ids[0..out.len]));
            try m.val.download(0, std.mem.sliceAsBytes(vals[0..out.len]));
            for (out, ids[0..out.len], vals[0..out.len]) |*o, id, val| o.* = try checkPick(id, val, @intCast(n));
            return;
        };
        for (out, positions, 0..) |*o, position, r| {
            const row = m.logits.ptr + r * n * 4;
            try head.argmax(x.c, x.s, row, 1, m.val.ptr, m.id.ptr);
            try x.s.synchronize();
            var id: u32 = undefined;
            var val: f32 = undefined;
            try m.id.download(0, std.mem.asBytes(&id));
            try m.val.download(0, std.mem.asBytes(&val));
            _ = try checkPick(id, val, @intCast(n)); // checks the top logit only, as `draw` does
            try ops.tobf(&x.c.ops, x.s, row, 1, n, n, x.bf.ptr, n, 0, n);
            const meta = [4]i32{ @intCast(position - 1), 0, 0, 0 };
            try x.rule.uploadAsync(0, std.mem.asBytes(&rule), x.s.handle);
            try x.meta.uploadAsync(0, std.mem.asBytes(&meta), x.s.handle);
            try x.draws.draw(x.s, x.bf.ptr, n, x.rule.ptr, x.meta.ptr, 0, x.out.ptr, 1, null, null);
            try x.s.synchronize();
            try x.out.download(0, std.mem.asBytes(o));
        }
    }

    /// One window: the pending row, the held drafts, then host drafts, through every visit at once; each row drawn at its position.
    fn runWindow(x: *Inkling, w: be.Window, o: be.Verified) !void {
        const m = if (x.mtp) |*m| m else return error.NoDraftHead;
        const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
        const rows = w.rows();
        if (w.positions.len != rows or o.sampled.len != rows or o.drafts.len != rows - 1) return error.PositionMismatch;
        if (w.held > l.n_drafts or (w.held > 0 and w.tokens.len > 0)) return error.NotHeld;
        var ids: [mtp_rows]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.held], l.drafts[0..w.held]);
        @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
        for (w.positions, 0..) |p, r| if (p != l.len + r + 1) return error.PositionMismatch;
        prof.begin(x.s.handle);
        x.arm(l, true);
        defer {
            x.c.window = false;
            x.c.recording = false;
        }
        try decode.forward(x.c, x.s, l.slot, ids[0..rows], l.len, x.c.sched.n - 1, .{});
        prof.at(prof.tag_head);
        try head.logits(x.c, x.s, x.c.flow.r.inp, rows, m.logits.ptr);
        try x.drawRows(m, l, w.positions, o.sampled);
        prof.window(@intCast(rows), 0);
        prof.end();
        @memcpy(o.drafts, ids[1..rows]);
        l.win_start = l.len;
        l.win_rows = @intCast(rows);
        l.len += rows;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        for (windows) |w| try admitWindow(w, x.rows);
        if (x.mtp == null) {
            for (windows, out) |w, o| {
                const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
                if (w.positions.len != window_rows) return error.PositionMismatch;
                o.sampled[0] = try x.step(l, w.pending, w.positions[0]);
            }
            return;
        }
        prof.roundBegin(x.s.handle);
        if (x.c.rec) |*r| try r.backup.copyFrom(0, x.c.cache.conv.ptr, x.c.cache.conv.len, x.s.handle);
        for (windows, out) |w, o| try x.runWindow(w, o);
    }

    /// A window keeps a prefix of its rows: the length, the conv states and the draft head's context follow it. The KV rows
    /// past the prefix are never read and the next forward writes over them.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            try admitPath(path, x.rows);
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (x.mtp == null) continue;
            const n: u32 = @intCast(path.len);
            if (n > l.win_rows) return error.WindowTooWide;
            l.len = l.win_start + n;
            prof.at(prof.tag_keep);
            prof.window(0, n);
            if (n < l.win_rows) {
                const r = x.c.rec.?;
                const table = r.sites.ptr + @as(u64, l.slot) * r.n_sites * @sizeOf(ops.RewindSite);
                try ops.rewind(&x.c.ops, x.s, table, r.n_sites, r.widest, r.backup.ptr, r.hist.ptr, 0, 0, n);
            }
            if (l.written < l.len) try x.absorb(l, @intCast(l.written - l.win_start), @intCast(l.len - l.written), l.written);
        }
    }

    /// Absorbs each stream's kept rows into the head's context, then holds `depth` drafts after its pending token.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| if (r.lanes != null or (x.mtp == null and r.depth != 0)) return error.NoDraftHead;
        const m = if (x.mtp) |*m| m else return;
        prof.at(prof.tag_draft);
        for (requests) |r| {
            const l = x.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            const at = r.position - 1; // the pending token's position: the kept length
            if (l.written < at) {
                if (l.written < l.win_start or at > l.win_start + l.win_rows) return error.PositionMismatch;
                try x.absorb(l, @intCast(l.written - l.win_start), @intCast(at - l.written), l.written);
            }
            l.n_drafts = 0;
            if (r.depth == 0) continue;
            const pending = if (r.first) |feed| try x.value(feed) else r.follow[r.follow.len - 1];
            var block: [most_drafts]u32 = undefined;
            try m.head.propose(x.s, l.slot, pending, at, &block, &l.chain);
            l.n_drafts = @min(r.depth, most_drafts);
            @memcpy(l.drafts[0..l.n_drafts], block[0..l.n_drafts]);
        }
    }

    /// Each held draft's chance of landing: the product of the head's probabilities along the chain.
    fn probabilities(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const x = self(ptr);
        const m = if (x.mtp) |*m| m else return false;
        if (!m.head.vcap) return false;
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (out.len > l.n_drafts) return false;
        var chance: f64 = 1;
        for (out, l.chain[0..out.len]) |*o, p| {
            chance *= p;
            o.* = chance;
        }
        return true;
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
    const drafts = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    return .{ .stream = undefined, .pending = 0, .held = 0, .tokens = drafts[0 .. rows - 1], .parents = parents, .positions = &.{}, .masks = masks };
}

test "window of one row without the draft head, of eight with it" {
    const mask = [_]u32{1};
    const cases = [_]struct { w: be.Window, err: anyerror }{
        .{ .w = window(2, null, &.{}), .err = error.WindowTooWide },
        .{ .w = window(4, &[_]i32{ -1, 0, 1, 2 }, &.{}), .err = error.WindowTooWide },
        .{ .w = window(1, &[_]i32{-1}, &.{}), .err = error.StructuresUnsupported },
        .{ .w = window(1, null, &mask), .err = error.StructuresUnsupported },
    };
    var x: Inkling = undefined;
    x.rows = window_rows;
    for (cases) |k| {
        var out: [1]be.Verified = .{.{ .sampled = &.{}, .drafts = &.{} }};
        try ts.expectError(k.err, Inkling.verify(&x, &.{k.w}, &out));
    }
    try admitWindow(window(4, null, &.{}), mtp_rows);
    try ts.expectError(error.WindowTooWide, admitWindow(window(mtp_rows + 1, null, &.{}), mtp_rows));
}

test "keep takes a prefix of the window's rows" {
    var x: Inkling = undefined;
    x.rows = window_rows;
    x.lanes = .empty;
    const w = window(1, null, &.{});
    try ts.expectError(error.WindowTooWide, Inkling.keep(&x, &.{w}, &.{&.{ 0, 1 }}));
    try ts.expectError(error.WindowTooWide, Inkling.keep(&x, &.{w}, &.{&.{1}}));
    try ts.expectError(error.EmptyPath, Inkling.keep(&x, &.{w}, &.{&.{}}));
    try admitPath(&.{ 0, 1, 2 }, mtp_rows);
    try ts.expectError(error.WindowTooWide, admitPath(&.{ 0, 2 }, mtp_rows));
    try ts.expectError(error.WindowTooWide, admitPath(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }, mtp_rows));
}

test "draft depth 0 only without the draft head" {
    var x: Inkling = undefined;
    x.mtp = null;
    const base: be.DraftRequest = .{ .stream = undefined, .follow = &.{}, .rows = null, .start = 0, .position = 1, .depth = 0 };
    var deep = base;
    deep.depth = 1;
    try Inkling.draft(&x, &.{base});
    try ts.expectError(error.NoDraftHead, Inkling.draft(&x, &.{ base, deep }));
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
    x.mtp = null;
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
