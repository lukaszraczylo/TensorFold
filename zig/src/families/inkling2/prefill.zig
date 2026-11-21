//! Chunked prefill of Inkling 2 Mini: a prompt goes through the 48 visits `chunk_rows` rows at a time (`decode.forward`
//! over a chunk), leaving the KV caches and conv states as a token-by-token run of the same ids does, so decode goes on.
const std = @import("std");
const cuda = @import("cuda");
const decode = @import("decode.zig");
const engine = @import("engine.zig");
const head = @import("head.zig");

/// Rows a chunk holds when the caller does not choose.
pub const default_chunk_rows = 512;
const bf16_bytes = 2;

/// The chunk sizes a prompt of `len` ids is cut into: `chunk_rows` each, the last one shorter.
pub const Chunks = struct {
    len: usize,
    chunk_rows: usize,
    at: usize = 0,

    pub fn next(it: *Chunks) ?struct { start: usize, rows: usize } {
        if (it.at >= it.len) return null;
        const rows = @min(it.chunk_rows, it.len - it.at);
        defer it.at += rows;
        return .{ .start = it.at, .rows = rows };
    }
};

/// One chunk of `ids.len <= chunk_rows` rows at positions `p0`.. of `stream` through all visits. `logits`, when given, is
/// the device address of fp32 [unpadded_vocab]: the head runs on the chunk's last row only.
pub fn chunk(c: *engine.Ctx, s: cuda.Stream, stream: u32, ids: []const u32, p0: u64, logits: ?u64, h: decode.Hooks) !void {
    if (ids.len == 0) return error.Inkling2EmptyChunk;
    try decode.forward(c, s, stream, ids, p0, c.sched.n - 1, h);
    const out = logits orelse return;
    const last_row = c.flow.r.inp + (ids.len - 1) * c.text.hidden * bf16_bytes;
    try head.logits(c, s, last_row, 1, out);
    try s.synchronize();
}

/// One chunk as `chunk` runs it, with the head's log-probs of the rows `want` selects into `out` (fp32 [want.count,
/// unpadded_vocab]) in place of the last row's logits.
pub fn chunkLogprobs(c: *engine.Ctx, s: cuda.Stream, stream: u32, ids: []const u32, p0: u64, want: head.RowRange, out: u64, h: decode.Hooks) !void {
    if (ids.len == 0) return error.Inkling2EmptyChunk;
    if (want.first + want.count > ids.len) return error.Inkling2RowsPastChunk;
    try decode.forward(c, s, stream, ids, p0, c.sched.n - 1, h);
    try head.logprobsRows(c, s, c.flow.r.inp, want.first + want.count, want.first, out);
    try s.synchronize();
}

/// A prompt from position 0 in chunks of `chunk_rows`; the final chunk gives `logits` (see `chunk`).
pub fn run(c: *engine.Ctx, s: cuda.Stream, stream: u32, ids: []const u32, chunk_rows: u32, logits: ?u64, h: decode.Hooks) !void {
    if (chunk_rows == 0 or chunk_rows > c.scratch.geometry.max_rows) return error.RowsExceedScratch;
    if (ids.len == 0) return error.Inkling2EmptyChunk;
    var it: Chunks = .{ .len = ids.len, .chunk_rows = chunk_rows };
    while (it.next()) |part| {
        const final = part.start + part.rows == ids.len;
        try chunk(c, s, stream, ids[part.start..][0..part.rows], part.start, if (final) logits else null, h);
    }
}

const ts = std.testing;

test "a prompt is cut into full chunks and a shorter last one" {
    var it: Chunks = .{ .len = 300, .chunk_rows = 128 };
    const want = [_][2]usize{ .{ 0, 128 }, .{ 128, 128 }, .{ 256, 44 } };
    for (want) |w| {
        const part = it.next().?;
        try ts.expectEqual(w[0], part.start);
        try ts.expectEqual(w[1], part.rows);
    }
    try ts.expect(it.next() == null);
}

test "chunks cover every id once for any chunk size" {
    for ([_]usize{ 1, 7, 64, 299, 300, 301, 512 }) |size| {
        var it: Chunks = .{ .len = 300, .chunk_rows = size };
        var next_start: usize = 0;
        var parts: usize = 0;
        while (it.next()) |part| {
            try ts.expectEqual(next_start, part.start);
            try ts.expect(part.rows >= 1 and part.rows <= size);
            next_start += part.rows;
            parts += 1;
        }
        try ts.expectEqual(@as(usize, 300), next_start);
        try ts.expectEqual((300 + size - 1) / size, parts);
    }
}

test "an empty prompt has no chunk" {
    var it: Chunks = .{ .len = 0, .chunk_rows = 512 };
    try ts.expect(it.next() == null);
}

test "run refuses a chunk size the scratch cannot hold before it touches the device" {
    var c: engine.Ctx = undefined;
    c.scratch.geometry.max_rows = 64;
    var d: cuda.Driver = undefined;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try ts.expectError(error.RowsExceedScratch, run(&c, s, 0, &.{1}, 65, null, .{}));
    try ts.expectError(error.RowsExceedScratch, run(&c, s, 0, &.{1}, 0, null, .{}));
    try ts.expectError(error.Inkling2EmptyChunk, run(&c, s, 0, &.{}, 64, null, .{}));
    try ts.expectError(error.Inkling2EmptyChunk, chunk(&c, s, 0, &.{}, 0, null, .{}));
}
