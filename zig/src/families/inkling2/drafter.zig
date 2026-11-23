//! Inkling 2 Mini's DSpark draft head (the checkpoint's mtp.safetensors, SGLang's inkling2_mtp.py): each committed position's
//! taps are mixed, normed and projected into the head's context caches; a block (the anchor token, then mask rows) runs the
//! head's layers over those caches and proposes the tokens after the anchor one by one through the Markov head.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const decode = @import("decode.zig");
const dk = @import("dspark_ops.zig");
const engine = @import("engine.zig");
const head = @import("head.zig");
const ops = @import("ops.zig");
const prefill4 = @import("prefill4.zig");
const weights = @import("weights.zig");

const platform = @import("sys");

const Checkpoint = core.checkpoint.Checkpoint;

pub const enable_env = "TF_INK2_MTP";
/// 1 stores the vocabulary fold and the layers' propose weights as int8 with fp16 block scales (drafts only).
pub const draft8_env = "TF_INK2_DRAFT8";

pub const file_name = "mtp.safetensors";
const bf16_bytes = 2;
const f32_bytes = 4;
/// The head's norm epsilon (mtp_config rms_norm_eps).
const eps: f32 = 1e-6;
/// Positions a global draft layer keeps (the last ones): the head's attention degrades, never the output.
const global_ring_cap = 8192;
/// Positions a local draft layer's ring holds: the window plus a block.
const local_ring = 2048;
/// Context rows a write projects at a time (a prompt chunk).
pub const write_rows = 512;
/// Rows of a block: the anchor and the proposals.
pub const block = 8;
pub const proposals = block - 1;
const max_layers = 8;
const rank_fixed = 256; // dspark.cu's fold and vocab kernels

const Layer = struct {
    local: bool,
    ring: u32,
    in_norm: cuda.DeviceBuffer,
    post_norm: cuda.DeviceBuffer,
    dflare: cuda.DeviceBuffer,
    q_norm: cuda.DeviceBuffer,
    k_norm: cuda.DeviceBuffer,
    k_norm_ctx: cuda.DeviceBuffer,
    wqkv: cuda.DeviceBuffer, // bf16 [q + 2 kv, hidden]
    wo: cuda.DeviceBuffer,
    wkv_ctx: cuda.DeviceBuffer, // bf16 [2 kv, hidden]: k_proj_ctx, then v_proj_ctx
    wgu: cuda.DeviceBuffer, // bf16 [2 inter, hidden]: gate, then up
    wd: cuda.DeviceBuffer,
    kc: cuda.DeviceBuffer, // bf16 [streams, ring, kv_heads, 128]
    vc: cuda.DeviceBuffer,
};

pub const Drafter = struct {
    c: *engine.Ctx,
    k: dk.Kernels,
    m: config.Mtp,
    streams: u32,
    draft8: bool,
    layers: [max_layers]Layer,
    alpha: cuda.DeviceBuffer, // fp32 [layers + 1, taps]: each layer's tap softmax, then the Markov head's
    norm: cuda.DeviceBuffer,
    mask: cuda.DeviceBuffer,
    mk_dflare: cuda.DeviceBuffer,
    w1: cuda.DeviceBuffer,
    rnn_w: cuda.DeviceBuffer,
    rnn_b: cuda.DeviceBuffer,
    fold: cuda.DeviceBuffer, // bf16 [vocab, rank]; draft8: int8 codes [vocab, rank], then fp16 scales [vocab, rank / 32]
    fbias: cuda.DeviceBuffer, // bf16 [vocab]
    frnn: cuda.DeviceBuffer, // bf16 [streams, hidden]
    // scratch, one call at a time
    resid: cuda.DeviceBuffer,
    h: cuda.DeviceBuffer,
    anchor_h: cuda.DeviceBuffer,
    tmp: cuda.DeviceBuffer, // fp32 [max(write_rows, block), widest projection]
    qkv: cuda.DeviceBuffer, // bf16 [max(write_rows, block), widest projection]
    q: cuda.DeviceBuffer,
    attn: cuda.DeviceBuffer,
    mix: cuda.DeviceBuffer,
    po: cuda.DeviceBuffer,
    pml: cuda.DeviceBuffer,
    base_f: cuda.DeviceBuffer,
    base_b: cuda.DeviceBuffer,
    ctx_rows: cuda.DeviceBuffer,
    state: cuda.DeviceBuffer,
    latent: cuda.DeviceBuffer,
    walk: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer,
    bytes: u64 = 0,

    fn dims(m: config.Mtp) struct { q: usize, kv: usize } {
        return .{ .q = @as(usize, m.q_heads) * m.head_dim, .kv = @as(usize, m.kv_heads) * m.head_dim };
    }

    fn tensor(ck: *Checkpoint, name: []const u8, shape: []const usize) ![]const u8 {
        return (try ck.expect(name, .bf16, shape)).bytes;
    }

    fn put(d: *const cuda.Driver, total: *u64, bytes: []const u8) !cuda.DeviceBuffer {
        var b = try cuda.DeviceBuffer.alloc(d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        total.* += bytes.len;
        return b;
    }

    fn zeroed(d: *const cuda.Driver, total: *u64, len: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(d, len);
        total.* += len;
        return b;
    }

    /// Loads `dir`/mtp.safetensors, sizes the context caches for `streams` streams of `context` positions and folds the Markov
    /// head's bias through the target's unembedding.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, c: *engine.Ctx, dir: []const u8, streams: u32, context: usize) !Drafter {
        if (!try platform.environment.read(bool, enable_env, false)) return error.MtpOff;
        const m = c.w.cfg.mtp orelse return error.NoDraftHead;
        const t = &c.text;
        const hidden: usize = t.hidden;
        const rank: usize = m.rank;
        const n_layers: usize = m.layers;
        const dm = dims(m);
        if (rank != rank_fixed or m.head_dim != 128 or m.block != block or n_layers > max_layers or m.taps_n > 8) return error.Inkling2DraftShape;
        if (block * (m.q_heads / m.kv_heads) > dk.max_q) return error.Inkling2DraftShape;
        const d = c.d;
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        defer ck.close();
        try ck.add(dir, file_name);

        var x: Drafter = undefined;
        x.c = c;
        x.m = m;
        x.streams = streams;
        x.draft8 = try platform.environment.read(bool, draft8_env, false);
        x.bytes = 0;
        x.k = try dk.Kernels.load(d);
        errdefer x.k.unload();
        var made: usize = 0;
        errdefer for (x.layers[0..made]) |*l| freeLayer(l);

        var name_buf: [96]u8 = undefined;
        for (0..n_layers) |i| {
            const l = &x.layers[i];
            const pre = try std.fmt.bufPrint(&name_buf, "layers.{d}.", .{i});
            var nb: [128]u8 = undefined;
            const at = struct {
                fn of(buf: []u8, p: []const u8, tail: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, "{s}{s}", .{ p, tail });
                }
            }.of;
            l.local = m.local[i];
            l.ring = if (l.local) local_ring else @intCast(@min(context, global_ring_cap));
            l.ring = @intCast(std.mem.alignForward(usize, l.ring, 256));
            l.in_norm = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "input_layernorm.weight"), &.{hidden}));
            errdefer l.in_norm.free();
            l.post_norm = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "post_attention_layernorm.weight"), &.{hidden}));
            errdefer l.post_norm.free();
            l.dflare = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "dflare_norm.weight"), &.{hidden}));
            errdefer l.dflare.free();
            l.q_norm = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "self_attn.q_norm.weight"), &.{m.head_dim}));
            errdefer l.q_norm.free();
            l.k_norm = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "self_attn.k_norm.weight"), &.{m.head_dim}));
            errdefer l.k_norm.free();
            l.k_norm_ctx = try put(d, &x.bytes, try tensor(&ck, try at(&nb, pre, "self_attn.k_norm_ctx.weight"), &.{m.head_dim}));
            errdefer l.k_norm_ctx.free();
            const wq = try tensor(&ck, try at(&nb, pre, "self_attn.q_proj.weight"), &.{ dm.q, hidden });
            const wk = try tensor(&ck, try at(&nb, pre, "self_attn.k_proj.weight"), &.{ dm.kv, hidden });
            const wv = try tensor(&ck, try at(&nb, pre, "self_attn.v_proj.weight"), &.{ dm.kv, hidden });
            l.wqkv = try joinW(gpa, d, &x.bytes, &.{ wq, wk, wv }, hidden, x.draft8);
            errdefer l.wqkv.free();
            l.wo = try joinW(gpa, d, &x.bytes, &.{try tensor(&ck, try at(&nb, pre, "self_attn.o_proj.weight"), &.{ hidden, dm.q })}, dm.q, x.draft8);
            errdefer l.wo.free();
            const wkc = try tensor(&ck, try at(&nb, pre, "self_attn.k_proj_ctx.weight"), &.{ dm.kv, hidden });
            const wvc = try tensor(&ck, try at(&nb, pre, "self_attn.v_proj_ctx.weight"), &.{ dm.kv, hidden });
            l.wkv_ctx = try join(d, &x.bytes, &.{ wkc, wvc });
            errdefer l.wkv_ctx.free();
            const wg = try tensor(&ck, try at(&nb, pre, "mlp.gate_proj.weight"), &.{ m.inter, hidden });
            const wu = try tensor(&ck, try at(&nb, pre, "mlp.up_proj.weight"), &.{ m.inter, hidden });
            l.wgu = try joinW(gpa, d, &x.bytes, &.{ wg, wu }, hidden, x.draft8);
            errdefer l.wgu.free();
            l.wd = try joinW(gpa, d, &x.bytes, &.{try tensor(&ck, try at(&nb, pre, "mlp.down_proj.weight"), &.{ hidden, m.inter })}, m.inter, x.draft8);
            errdefer l.wd.free();
            const cache_bytes = @as(usize, streams) * l.ring * dm.kv * bf16_bytes;
            l.kc = try zeroed(d, &x.bytes, cache_bytes);
            errdefer l.kc.free();
            l.vc = try zeroed(d, &x.bytes, cache_bytes);
            made += 1;
        }
        errdefer freeBuffers(&x, made);

        x.norm = try put(d, &x.bytes, try tensor(&ck, "norm.weight", &.{hidden}));
        x.mask = try put(d, &x.bytes, try tensor(&ck, "mask_embed", &.{hidden}));
        x.mk_dflare = try put(d, &x.bytes, try tensor(&ck, "markov_head.dflare_norm.weight", &.{hidden}));
        const vocab_rows: usize = t.vocab;
        x.w1 = try put(d, &x.bytes, try tensor(&ck, "markov_head.markov_w1.weight", &.{ vocab_rows, rank }));
        x.rnn_w = try put(d, &x.bytes, try tensor(&ck, "markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * rank + 2 * hidden }));
        x.rnn_b = try put(d, &x.bytes, try tensor(&ck, "markov_head.rnn_proj.bias", &.{3 * rank}));
        const hout_w = try tensor(&ck, "markov_head.hidden_out.weight", &.{ hidden, rank });
        const hout_b = try tensor(&ck, "markov_head.hidden_out.bias", &.{hidden});

        const taps: usize = m.taps_n;
        const fuse_t = try ck.get("fuse_weight");
        if (fuse_t.dtype != .bf16 or fuse_t.rank != 2 or fuse_t.shape[0] < n_layers + 1 or fuse_t.shape[1] != taps) return error.Inkling2DraftShape;
        const fuse = fuse_t.bytes;
        var alpha_host: [(max_layers + 1) * 8]f32 = undefined;
        softmaxRows(fuse, n_layers + 1, taps, &alpha_host);
        x.alpha = try put(d, &x.bytes, std.mem.sliceAsBytes(alpha_host[0 .. (n_layers + 1) * taps]));

        const vocab: usize = t.unpadded_vocab;
        x.fold = try zeroed(d, &x.bytes, if (x.draft8) weights.q8Bytes(vocab, rank) else vocab * rank * bf16_bytes);
        x.fbias = try zeroed(d, &x.bytes, vocab * bf16_bytes);
        x.frnn = try zeroed(d, &x.bytes, streams * hidden * bf16_bytes);

        const wide = @max(dm.q + 2 * dm.kv, 2 * m.inter);
        const rows = @max(write_rows, block);
        x.resid = try zeroed(d, &x.bytes, block * hidden * bf16_bytes);
        x.h = try zeroed(d, &x.bytes, block * hidden * bf16_bytes);
        x.anchor_h = try zeroed(d, &x.bytes, hidden * bf16_bytes);
        x.tmp = try zeroed(d, &x.bytes, rows * wide * f32_bytes);
        x.qkv = try zeroed(d, &x.bytes, rows * (dm.q + 2 * dm.kv) * bf16_bytes);
        x.q = try zeroed(d, &x.bytes, block * @max(dm.q, m.inter) * bf16_bytes);
        x.attn = try zeroed(d, &x.bytes, block * dm.q * bf16_bytes);
        x.mix = try zeroed(d, &x.bytes, write_rows * hidden * bf16_bytes);
        const chunks = global_ring_cap / dk.chunk + 2;
        x.po = try zeroed(d, &x.bytes, chunks * block * m.q_heads * m.head_dim * f32_bytes);
        x.pml = try zeroed(d, &x.bytes, chunks * block * m.q_heads * 2 * f32_bytes);
        x.base_f = try zeroed(d, &x.bytes, proposals * vocab * f32_bytes);
        x.base_b = try zeroed(d, &x.bytes, proposals * vocab * bf16_bytes);
        x.ctx_rows = try zeroed(d, &x.bytes, proposals * 3 * rank * f32_bytes);
        x.state = try zeroed(d, &x.bytes, rank * bf16_bytes);
        x.latent = try zeroed(d, &x.bytes, rank * bf16_bytes);
        x.walk = try zeroed(d, &x.bytes, block * 4);
        x.picks = try zeroed(d, &x.bytes, 16);
        x.logits = try zeroed(d, &x.bytes, vocab * f32_bytes);
        try x.foldHead(gpa, hout_w, hout_b);
        return x;
    }

    fn join(d: *const cuda.Driver, total: *u64, parts: []const []const u8) !cuda.DeviceBuffer {
        var n: usize = 0;
        for (parts) |p| n += p.len;
        var b = try cuda.DeviceBuffer.alloc(d, n);
        errdefer b.free();
        var at: usize = 0;
        for (parts) |p| {
            try b.upload(at, p);
            at += p.len;
        }
        total.* += n;
        return b;
    }

    /// `join` of bf16 weights whose rows are `k` long, stored int8 with fp16 block scales when `q8` (weights.quantQ8's layout).
    fn joinW(gpa: std.mem.Allocator, d: *const cuda.Driver, total: *u64, parts: []const []const u8, k: usize, q8: bool) !cuda.DeviceBuffer {
        if (!q8) return join(d, total, parts);
        var n: usize = 0;
        for (parts) |p| n += p.len;
        const host = try gpa.alloc(u8, n);
        defer gpa.free(host);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(host[at..][0..p.len], p);
            at += p.len;
        }
        const rows = n / (k * bf16_bytes);
        if (k % weights.q8_block != 0 or rows * k * bf16_bytes != n) return error.Inkling2DraftShape;
        const out = try gpa.alloc(u8, weights.q8Bytes(rows, k));
        defer gpa.free(out);
        weights.quantQ8(host, rows, k, out);
        return put(d, total, out);
    }

    fn softmaxRows(raw: []const u8, rows: usize, cols: usize, out: []f32) void {
        const vals = std.mem.bytesAsSlice(u16, raw);
        for (0..rows) |r| {
            var mx: f32 = -std.math.floatMax(f32);
            for (0..cols) |j| mx = @max(mx, bf16ToF32(vals[r * cols + j]));
            var sum: f32 = 0;
            for (0..cols) |j| {
                const e = @exp(bf16ToF32(vals[r * cols + j]) - mx);
                out[r * cols + j] = e;
                sum += e;
            }
            for (0..cols) |j| out[r * cols + j] /= sum;
        }
    }

    fn bf16ToF32(v: u16) f32 {
        return @bitCast(@as(u32, v) << 16);
    }

    /// fold[v] = unembed[v] . hidden_out (per rank column) and fbias[v] = unembed[v] . hidden_out.bias, rounded to bf16.
    fn foldHead(x: *Drafter, gpa: std.mem.Allocator, hout_w: []const u8, hout_b: []const u8) !void {
        const c = x.c;
        const hidden: usize = c.text.hidden;
        const rank: usize = x.m.rank;
        const w = c.w.unembed;
        const vocab: usize = c.text.unpadded_vocab;
        const stored: usize = w.shape[0];
        // x rows: hidden_out^T (rank rows of hidden), then its bias
        const rows_host = try gpa.alloc(u16, (rank + 1) * hidden);
        defer gpa.free(rows_host);
        const src = std.mem.bytesAsSlice(u16, hout_w);
        for (0..hidden) |col| for (0..rank) |r| {
            rows_host[r * hidden + col] = src[col * rank + r];
        };
        @memcpy(rows_host[rank * hidden ..][0..hidden], std.mem.bytesAsSlice(u16, hout_b));
        var rows_dev = try cuda.DeviceBuffer.alloc(c.d, rows_host.len * bf16_bytes);
        defer rows_dev.free();
        try rows_dev.upload(0, std.mem.sliceAsBytes(rows_host));
        const tile_max = engine.dq_tile_rows;
        var out = try cuda.DeviceBuffer.alloc(c.d, (rank + 1) * tile_max * f32_bytes);
        defer out.free();
        var s = try cuda.Stream.init(c.d, true);
        defer s.deinit();
        var row0: usize = 0;
        while (row0 < vocab) : (row0 += tile_max) {
            const tile = @min(tile_max, vocab - row0);
            var wptr: u64 = undefined;
            if (w.q8) {
                try ops.q8ToBf16(&c.ops, s, w.ptr, stored, hidden, row0, tile, c.dq.ptr);
                wptr = c.dq.ptr;
            } else wptr = w.ptr + row0 * hidden * bf16_bytes;
            try c.dense.rows(s, rows_dev.ptr, hidden, wptr, out.ptr, tile_max, rank + 1, tile, hidden);
            if (x.draft8) {
                try dk.foldTileQ8(&x.k, s, out.ptr, tile, tile_max, row0, rank, x.fold.ptr, x.fold.ptr + vocab * rank, x.fbias.ptr);
            } else try dk.foldTile(&x.k, s, out.ptr, tile, tile_max, row0, rank, x.fold.ptr, x.fbias.ptr);
        }
        try s.synchronize();
    }

    fn freeLayer(l: *Layer) void {
        inline for (.{ "in_norm", "post_norm", "dflare", "q_norm", "k_norm", "k_norm_ctx", "wqkv", "wo", "wkv_ctx", "wgu", "wd", "kc", "vc" }) |f| @field(l, f).free();
    }

    fn freeBuffers(x: *Drafter, made: usize) void {
        _ = x;
        _ = made;
    }

    pub fn deinit(x: *Drafter) void {
        inline for (.{ "alpha", "norm", "mask", "mk_dflare", "w1", "rnn_w", "rnn_b", "fold", "fbias", "frnn", "resid", "h", "anchor_h", "tmp", "qkv", "q", "attn", "mix", "po", "pml", "base_f", "base_b", "ctx_rows", "state", "latent", "walk", "picks", "logits" }) |f| @field(x, f).free();
        for (x.layers[0..x.m.layers]) |*l| freeLayer(l);
        x.k.unload();
        x.* = undefined;
    }

    /// out fp32 [m, n] = x bf16 [m, k] times w bf16 [n, k] transposed.
    fn gemm(x: *const Drafter, s: cuda.Stream, a: u64, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        if (x.c.pf4) |*st| if (st.dense and m >= prefill4.prompt_from and k % 16 == 0 and n % 8 == 0 and a % 4 == 0 and w % 4 == 0) return prefill4.denseRows(&st.k, s, a, k, w, out, n, m, n, k);
        try x.c.dense.rows(s, a, k, w, out, n, m, n, k);
    }

    /// bf16 out = round(x . w^T).
    fn gemmBf(x: *const Drafter, s: cuda.Stream, a: u64, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        try x.gemm(s, a, w, x.tmp.ptr, m, n, k);
        try ops.tobf(&x.c.ops, s, x.tmp.ptr, m, n, n, out, n, 0, n);
    }

    /// `gemm` / `gemmBf` over a `joinW` weight.
    fn gemmW(x: *const Drafter, s: cuda.Stream, a: u64, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        if (!x.draft8) return x.gemm(s, a, w, out, m, n, k);
        const seg: ops.DenseSeg = .{ .x = a, .w = w, .out = out, .n = @intCast(n), .block0 = 0 };
        try ops.dense1Q8Rows(&x.c.ops, s, &.{seg}, k, true, 0, m);
    }

    fn gemmBfW(x: *const Drafter, s: cuda.Stream, a: u64, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        if (!x.draft8) return x.gemmBf(s, a, w, out, m, n, k);
        const seg: ops.DenseSeg = .{ .x = a, .w = w, .out = out, .n = @intCast(n), .block0 = 0 };
        try ops.dense1Q8Rows(&x.c.ops, s, &.{seg}, k, false, 0, m);
    }

    /// A stream's context at `at` positions, in image order: each layer's last `min(at, ring)` K rows, then its V rows
    /// (position p at slot p % ring), then the Markov head's recurrent input. `f` gets each run's device address, its offset
    /// in the image and its length.
    pub fn each(x: *const Drafter, stream: u32, at: u32, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: u64, len: u64) anyerror!void) !void {
        const row = @as(u64, x.m.kv_heads) * x.m.head_dim * bf16_bytes;
        var off: u64 = 0;
        for (x.layers[0..x.m.layers]) |*l| for ([_]u64{ l.kc.ptr, l.vc.ptr }) |base| {
            const keep = @min(at, l.ring);
            var p: u64 = at - keep;
            while (p < at) {
                const slot = p % l.ring;
                const run = @min(at - p, l.ring - slot);
                try f(ctx, base + x.ringBase(l, stream) + slot * row, off, run * row);
                off += run * row;
                p += run;
            }
        };
        const rnn = @as(u64, x.c.text.hidden) * bf16_bytes;
        try f(ctx, x.frnn.ptr + @as(u64, stream) * rnn, off, rnn);
    }

    /// The bytes `each` copies for `at` positions.
    pub fn snapBytes(x: *const Drafter, at: u32) u64 {
        const row = @as(u64, x.m.kv_heads) * x.m.head_dim * bf16_bytes;
        var n: u64 = @as(u64, x.c.text.hidden) * bf16_bytes;
        for (x.layers[0..x.m.layers]) |*l| n += 2 * @min(at, l.ring) * row;
        return n;
    }

    fn ringBase(x: *const Drafter, l: *const Layer, stream: u32) u64 {
        const row = @as(u64, x.m.kv_heads) * x.m.head_dim * bf16_bytes;
        return @as(u64, stream) * l.ring * row;
    }

    /// The context caches' rows for `rows` positions from `pos0` of `stream`, from their taps (bf16 [rows, taps, hidden] at `taps`).
    pub fn writeContext(x: *Drafter, s: cuda.Stream, stream: u32, taps: u64, rows: usize, pos0: u64) !void {
        if (rows == 0 or rows > write_rows) return error.RowsExceedScratch;
        const c = x.c;
        const hidden: usize = c.text.hidden;
        const dm = dims(x.m);
        const n_taps: usize = x.m.taps_n;
        for (x.layers[0..x.m.layers], 0..) |*l, i| {
            try dk.mix(&x.k, s, taps, x.alpha.ptr + i * n_taps * f32_bytes, l.dflare.ptr, x.mix.ptr, rows, n_taps, hidden, eps);
            try x.gemmBf(s, x.mix.ptr, l.wkv_ctx.ptr, x.qkv.ptr, rows, 2 * dm.kv, hidden);
            const base = x.ringBase(l, stream);
            try dk.qk(&x.k, s, .{
                .x = x.qkv.ptr,
                .qw = l.k_norm_ctx.ptr,
                .kw = l.k_norm_ctx.ptr,
                .q = 0,
                .kc = l.kc.ptr + base,
                .vc = l.vc.ptr + base,
                .ld = @intCast(2 * dm.kv),
                .hq = 0,
                .hk = @intCast(x.m.kv_heads),
                .ring = @intCast(l.ring),
                .pos0 = @intCast(pos0),
                .eps = eps,
                .theta = x.m.rope_theta,
            }, rows);
        }
    }

    /// The Markov head's context row of `stream` from tap row `row` of `taps`.
    pub fn setRnnContext(x: *Drafter, s: cuda.Stream, stream: u32, taps: u64, row: usize) !void {
        const hidden: usize = x.c.text.hidden;
        const n_taps: usize = x.m.taps_n;
        const alpha = x.alpha.ptr + @as(u64, x.m.layers) * n_taps * f32_bytes;
        try dk.mix(&x.k, s, taps + row * n_taps * hidden * bf16_bytes, alpha, x.mk_dflare.ptr, x.frnn.ptr + @as(u64, stream) * hidden * bf16_bytes, 1, n_taps, hidden, eps);
    }

    /// The `proposals` tokens after `anchor` (position `pos`, its embedding drawn by the target) into `out`.
    pub fn propose(x: *Drafter, s: cuda.Stream, stream: u32, anchor: u32, pos: u64, out: *[proposals]u32) !void {
        const c = x.c;
        const t = &c.text;
        const hidden: usize = t.hidden;
        const dm = dims(x.m);
        const vocab: usize = t.unpadded_vocab;
        const rank: usize = x.m.rank;
        const k = &x.k;
        try decode.embedTokens(c, s, &.{anchor}, x.anchor_h.ptr);
        try dk.blockInit(k, s, x.anchor_h.ptr, x.mask.ptr, x.resid.ptr, block, hidden);
        try dk.addNorm(k, s, x.resid.ptr, 0, 0, 0, x.layers[0].in_norm.ptr, x.h.ptr, block, hidden, eps);
        for (x.layers[0..x.m.layers], 0..) |*l, i| {
            try x.gemmBfW(s, x.h.ptr, l.wqkv.ptr, x.qkv.ptr, block, dm.q + 2 * dm.kv, hidden);
            const base = x.ringBase(l, stream);
            try dk.qk(k, s, .{
                .x = x.qkv.ptr,
                .qw = l.q_norm.ptr,
                .kw = l.k_norm.ptr,
                .q = x.q.ptr,
                .kc = l.kc.ptr + base,
                .vc = l.vc.ptr + base,
                .ld = @intCast(dm.q + 2 * dm.kv),
                .hq = @intCast(x.m.q_heads),
                .hk = @intCast(x.m.kv_heads),
                .ring = @intCast(l.ring),
                .pos0 = @intCast(pos),
                .eps = eps,
                .theta = x.m.rope_theta,
            }, block);
            const hi: usize = @intCast(pos + block - 1);
            const span: usize = if (l.local) x.m.window else l.ring;
            const lo: usize = if (hi + 1 > span) hi + 1 - span else 0;
            try dk.attend(k, s, .{
                .q = x.q.ptr,
                .kc = l.kc.ptr + base,
                .vc = l.vc.ptr + base,
                .po = x.po.ptr,
                .pml = x.pml.ptr,
                .T = block,
                .hq = @intCast(x.m.q_heads),
                .hk = @intCast(x.m.kv_heads),
                .ring = @intCast(l.ring),
                .lo = @intCast(lo),
                .hi = @intCast(hi),
                .c0 = 0,
                .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(x.m.head_dim))),
            }, x.attn.ptr);
            try x.gemmW(s, x.attn.ptr, l.wo.ptr, x.tmp.ptr, block, hidden, dm.q);
            try dk.addNorm(k, s, x.resid.ptr, x.tmp.ptr, 1, 0, l.post_norm.ptr, x.h.ptr, block, hidden, eps);
            try x.gemmBfW(s, x.h.ptr, l.wgu.ptr, x.qkv.ptr, block, 2 * x.m.inter, hidden);
            try dk.silu(k, s, x.qkv.ptr, x.q.ptr, block, x.m.inter);
            try x.gemmW(s, x.q.ptr, l.wd.ptr, x.tmp.ptr, block, hidden, x.m.inter);
            const next = if (i + 1 < x.m.layers) x.layers[i + 1].in_norm.ptr else x.norm.ptr;
            try dk.addNorm(k, s, x.resid.ptr, x.tmp.ptr, 1, 0, next, x.h.ptr, block, hidden, eps);
        }
        // base logits of rows 1.. through the target's head, rounded to bf16
        const hrows = x.h.ptr + hidden * bf16_bytes;
        const was = c.window;
        c.window = true;
        defer c.window = was;
        try head.logitsRaw(c, s, hrows, proposals, x.base_f.ptr);
        try ops.tobf(&c.ops, s, x.base_f.ptr, proposals, vocab, vocab, x.base_b.ptr, vocab, 0, vocab);
        try dk.ctx(k, s, hrows, x.frnn.ptr + @as(u64, stream) * hidden * bf16_bytes, x.rnn_w.ptr, x.rnn_b.ptr, x.ctx_rows.ptr, proposals, hidden, rank);
        try x.state.fill8(0, s.handle);
        const first = [1]i32{@intCast(anchor)};
        try x.walk.uploadAsync(0, std.mem.sliceAsBytes(&first), s.handle);
        for (0..proposals) |j| {
            try dk.rnn(k, s, x.ctx_rows.ptr + j * 3 * rank * f32_bytes, x.rnn_w.ptr, x.w1.ptr, x.walk.ptr, j, x.state.ptr, x.latent.ptr, rank, hidden);
            const base_row = x.base_b.ptr + j * vocab * bf16_bytes;
            if (x.draft8) {
                try dk.vocabQ8(k, s, base_row, x.latent.ptr, x.fold.ptr, x.fold.ptr + vocab * rank, x.fbias.ptr, t.logit_cap, vocab, vocab, x.logits.ptr);
            } else try dk.vocab(k, s, base_row, x.latent.ptr, x.fold.ptr, x.fbias.ptr, t.logit_cap, vocab, vocab, x.logits.ptr);
            try ops.top1(&c.ops, s, x.logits.ptr, 1, vocab, vocab, 0, x.picks.ptr, x.picks.ptr + 4);
            try dk.pick(k, s, x.picks.ptr, 1, x.walk.ptr, j);
        }
        var got: [block]i32 = undefined;
        try x.walk.downloadAsync(0, std.mem.sliceAsBytes(&got), s.handle);
        try s.synchronize();
        for (out, got[1..]) |*o, g| o.* = std.math.cast(u32, g) orelse return error.Inkling2DraftToken;
    }
};
