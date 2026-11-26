//! Inkling 2 Mini's DFlash2 draft model (--drafter <folder of incoai/Inkling-2-Mini-DFlash2>, Apache-2.0, read unmodified): the
//! target's tapped rows (visits 5, 23, 41 and the final norm, the DSpark head's taps) go through fc and the hidden norm into
//! five layers' context rings; one block (the pending token and seven masks) runs the layers' two-tap convs, non-causal
//! attention and SwiGLU, the target's head gives each row's 16 best tokens, and the candidate selector picks the greedy path.
//! Adapted from MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold patch 0018 (Apache-2.0) and this fork's GLM DFlash2 (glm_dflash.cu).
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const decode = @import("decode.zig");
const engine = @import("engine.zig");
const head = @import("head.zig");
const platform = @import("sys");
const ops = @import("ops.zig");
const weights = @import("weights.zig");

const Checkpoint = core.checkpoint.Checkpoint;

/// Target visits the drafter taps, in the order fc's input concatenates them (dflash_config.target_layer_ids).
pub const taps = [_]u32{ 5, 23, 41, 48 };
pub const hidden = 2304;
pub const layers_n = 5;
const q_heads = 32;
const kv_heads = 8;
const head_dim = 128;
const q_width = q_heads * head_dim;
const kv_width = kv_heads * head_dim;
const qkv_width = q_width + 2 * kv_width;
pub const inter = 6912;
const group = 16;
const dyn_width = 2 * 2 * (hidden / group);
pub const rank = 256;
const top = 16;
pub const block = 8;
pub const proposals = block - 1;
pub const window = 2048;
/// Context rows one `writeContext` takes (a prompt chunk).
pub const write_rows = 512;
pub const mask_token: u32 = 200056;
const theta: f32 = 10000;
const eps: f32 = 1e-6;
const cap: f32 = 30;
const chunk_cols = 4096;
const attn_chunks = 16;
const bf16_bytes = 2;
const f32_bytes = 4;
pub const ring_bytes: usize = layers_n * 2 * window * kv_width * bf16_bytes;
const walk_bytes = 2 * block * 4;
pub const file_name = "model.safetensors";
/// 0 keeps the block pass's linears in bf16; on (the default) they are 8-bit with block scales (drafts only, verify unchanged).
pub const layers8_env = "TF_INK2_DRAFT_LAYERS8";
/// 0 turns the calibrated lone-stream chances off (the depth rule prices the drafts); on by default.
pub const chances_env = "TF_INK2_DFLASH2_CHANCES";

const Layer = struct {
    qkv: cuda.DeviceBuffer, // bf16 [q | k | v][hidden]
    kv: cuda.DeviceBuffer, // bf16 [k | v][hidden]
    o: cuda.DeviceBuffer,
    gu: cuda.DeviceBuffer, // bf16 [gate | up][hidden]
    down: cuda.DeviceBuffer,
    a_proj: cuda.DeviceBuffer, // attention_conv.kernel_projection
    m_proj: cuda.DeviceBuffer, // mlp_conv.kernel_projection
    in_norm: cuda.DeviceBuffer,
    post_norm: cuda.DeviceBuffer,
    q_norm: cuda.DeviceBuffer,
    k_norm: cuda.DeviceBuffer,
    a_base: cuda.DeviceBuffer, // bf16 [2][2][hidden]
    m_base: cuda.DeviceBuffer,
};

const Kernels = struct {
    mod: cuda.Module,
    norm: cuda.Function,
    conv: cuda.Function,
    swiglu: cuda.Function,
    block_qkv: cuda.Function,
    ctx_kv: cuda.Function,
    attn_part: cuda.Function,
    attn_merge: cuda.Function,
    top16_part: cuda.Function,
    top16_merge: cuda.Function,
    select: cuda.Function,

    fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dflash);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (.{ "norm", "conv", "swiglu", "block_qkv", "ctx_kv", "attn_part", "attn_merge", "top16_part", "top16_merge", "select" }) |name| {
            @field(k, name) = try mod.function("tf_ink2d_" ++ name);
        }
        return k;
    }
};

/// A stream's ring state on the host: the first position the ring holds for this sequence and the end of what it absorbed.
const Lane = struct { from: u64 = 0, end: u64 = 0 };

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

pub const Dflash2 = struct {
    c: *engine.Ctx,
    k: Kernels,
    streams: u32,
    vocab: usize,
    layers: [layers_n]Layer,
    fc: cuda.DeviceBuffer,
    hproj: cuda.DeviceBuffer,
    hnorm: cuda.DeviceBuffer,
    norm: cuda.DeviceBuffer,
    pred: cuda.DeviceBuffer, // bf16 [vocab][rank]
    succ: cuda.DeviceBuffer,
    rings: cuda.DeviceBuffer, // [streams][ring_bytes]
    walks: cuda.DeviceBuffer, // per stream: int32 [block] path, fp32 [block] chances
    walk_table: cuda.DeviceBuffer, // u64 [streams]: each walk's address
    lane: []Lane,
    gpa: std.mem.Allocator,
    // scratch, one call at a time
    tmp: cuda.DeviceBuffer, // fp32 [max(write_rows, block)][widest projection]
    c1: cuda.DeviceBuffer,
    ctx: cuda.DeviceBuffer,
    ckv: cuda.DeviceBuffer,
    x: cuda.DeviceBuffer,
    y: cuda.DeviceBuffer,
    y2: cuda.DeviceBuffer,
    dyn: cuda.DeviceBuffer,
    qkv: cuda.DeviceBuffer,
    q: cuda.DeviceBuffer,
    kb: cuda.DeviceBuffer,
    vb: cuda.DeviceBuffer,
    att: cuda.DeviceBuffer,
    a: cuda.DeviceBuffer,
    gu: cuda.DeviceBuffer,
    act: cuda.DeviceBuffer,
    logits: cuda.DeviceBuffer, // fp32 [proposals][vocab]
    pv: cuda.DeviceBuffer,
    pi: cuda.DeviceBuffer,
    cand: cuda.DeviceBuffer, // int32 [proposals][2 * top]
    g: cuda.DeviceBuffer, // fp32 [block][rank]
    pm: cuda.DeviceBuffer,
    pl: cuda.DeviceBuffer,
    pacc: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer, // int32 [block]
    bpos: cuda.DeviceBuffer,
    ring_ptr: cuda.DeviceBuffer, // u64, i64, i64: this call's ring, start and first position
    start: cuda.DeviceBuffer,
    from: cuda.DeviceBuffer,
    q8: bool,
    vcap: bool = false, // chances on: `propose` reports each draft's own probability
    vcap_floor: ?f64 = null,
    bytes: u64 = 0,

    fn tensor(ck: *Checkpoint, name: []const u8, shape: []const usize) ![]const u8 {
        return (ck.expect(name, .bf16, shape) catch |e| {
            std.log.err("Inkling 2 Mini DFlash2: {s} is missing or not bf16 of the expected shape ({s})", .{ name, @errorName(e) });
            return error.DFlash2Tensor;
        }).bytes;
    }

    fn put(x: *Dflash2, parts: []const []const u8) !cuda.DeviceBuffer {
        var n: usize = 0;
        for (parts) |p| n += p.len;
        var b = try cuda.DeviceBuffer.alloc(x.c.d, n);
        errdefer b.free();
        var at: usize = 0;
        for (parts) |p| {
            try b.upload(at, p);
            at += p.len;
        }
        x.bytes += n;
        return b;
    }

    /// A block-pass weight: bf16 rows joined, or 8-bit with block scales (`k` inputs a row).
    fn putW(x: *Dflash2, parts: []const []const u8, k: usize) !cuda.DeviceBuffer {
        if (!x.q8) return x.put(parts);
        var n: usize = 0;
        for (parts) |p| n += p.len;
        const host = try x.gpa.alloc(u8, n);
        defer x.gpa.free(host);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(host[at..][0..p.len], p);
            at += p.len;
        }
        const rows = n / (k * bf16_bytes);
        if (k % weights.q8_block != 0) return error.UnsupportedDFlash2Config;
        const out = try x.gpa.alloc(u8, weights.q8Bytes(rows, k));
        defer x.gpa.free(out);
        weights.quantQ8(host, rows, k, out);
        return x.put(&.{out});
    }

    fn zeroed(x: *Dflash2, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(x.c.d, @max(n, 256));
        x.bytes += @max(n, 256);
        return b;
    }

    fn layerTensor(ck: *Checkpoint, i: usize, suffix: []const u8, shape: []const usize) ![]const u8 {
        var b: [96]u8 = undefined;
        return tensor(ck, try std.fmt.bufPrint(&b, "layers.{d}.{s}", .{ i, suffix }), shape);
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, c: *engine.Ctx, dir: []const u8, streams: u32) !Dflash2 {
        try admit(gpa, io, dir);
        const t = &c.text;
        if (t.hidden != hidden or t.unpadded_vocab <= mask_token) return error.UnsupportedDFlash2Config;
        const m = c.w.cfg.mtp orelse return error.NoDraftHead;
        if (m.taps_n != taps.len) return error.UnsupportedDFlash2Config;
        for (taps, m.taps[0..taps.len]) |want, got| if (want != got) return error.UnsupportedDFlash2Config;
        var ck: Checkpoint = .{ .gpa = gpa, .io = io };
        defer ck.close();
        ck.add(dir, file_name) catch |e| {
            std.log.err("Inkling 2 Mini DFlash2: cannot open {s}/{s} ({s})", .{ dir, file_name, @errorName(e) });
            return error.DFlash2MissingFile;
        };
        var x: Dflash2 = undefined;
        x.c = c;
        x.gpa = gpa;
        x.streams = streams;
        x.vocab = t.unpadded_vocab;
        x.bytes = 0;
        x.k = try Kernels.load(c.d);
        x.q8 = try platform.environment.read(bool, layers8_env, true);
        x.vcap = try platform.environment.read(bool, chances_env, true);
        x.vcap_floor = null;
        x.lane = try gpa.alloc(Lane, streams);
        @memset(x.lane, .{});
        try x.loadWeights(&ck);
        try x.scratch();
        return x;
    }

    fn loadWeights(x: *Dflash2, ck: *Checkpoint) !void {
        for (&x.layers, 0..) |*l, i| {
            const q = try layerTensor(ck, i, "self_attn.q_proj.weight", &.{ q_width, hidden });
            const k = try layerTensor(ck, i, "self_attn.k_proj.weight", &.{ kv_width, hidden });
            const v = try layerTensor(ck, i, "self_attn.v_proj.weight", &.{ kv_width, hidden });
            const gate = try layerTensor(ck, i, "mlp.gate_proj.weight", &.{ inter, hidden });
            const up = try layerTensor(ck, i, "mlp.up_proj.weight", &.{ inter, hidden });
            l.qkv = try x.putW(&.{ q, k, v }, hidden);
            l.kv = try x.put(&.{ k, v });
            l.o = try x.putW(&.{try layerTensor(ck, i, "self_attn.o_proj.weight", &.{ hidden, q_width })}, q_width);
            l.gu = try x.putW(&.{ gate, up }, hidden);
            l.down = try x.putW(&.{try layerTensor(ck, i, "mlp.down_proj.weight", &.{ hidden, inter })}, inter);
            l.a_proj = try x.putW(&.{try layerTensor(ck, i, "attention_conv.kernel_projection.weight", &.{ dyn_width, hidden })}, hidden);
            l.m_proj = try x.putW(&.{try layerTensor(ck, i, "mlp_conv.kernel_projection.weight", &.{ dyn_width, hidden })}, hidden);
            l.in_norm = try x.put(&.{try layerTensor(ck, i, "input_layernorm.weight", &.{hidden})});
            l.post_norm = try x.put(&.{try layerTensor(ck, i, "post_attention_layernorm.weight", &.{hidden})});
            l.q_norm = try x.put(&.{try layerTensor(ck, i, "self_attn.q_norm.weight", &.{head_dim})});
            l.k_norm = try x.put(&.{try layerTensor(ck, i, "self_attn.k_norm.weight", &.{head_dim})});
            l.a_base = try x.put(&.{try layerTensor(ck, i, "attention_conv.base_kernel", &.{ 2, 2, hidden })});
            l.m_base = try x.put(&.{try layerTensor(ck, i, "mlp_conv.base_kernel", &.{ 2, 2, hidden })});
        }
        x.fc = try x.put(&.{try tensor(ck, "fc.weight", &.{ hidden, taps.len * hidden })});
        x.hproj = try x.put(&.{try tensor(ck, "candidate_selector.hidden_projection.weight", &.{ rank, hidden })});
        x.hnorm = try x.put(&.{try tensor(ck, "hidden_norm.weight", &.{hidden})});
        x.norm = try x.put(&.{try tensor(ck, "norm.weight", &.{hidden})});
        const book_rows = x.c.w.cfg.text.vocab;
        const pred = try tensor(ck, "candidate_selector.predecessor_codebook", &.{ book_rows, rank });
        x.pred = try x.put(&.{pred[0 .. x.vocab * rank * bf16_bytes]});
        const succ = try tensor(ck, "candidate_selector.successor_codebook", &.{ book_rows, rank });
        x.succ = try x.put(&.{succ[0 .. x.vocab * rank * bf16_bytes]});
    }

    fn scratch(x: *Dflash2) !void {
        const rows = @max(write_rows, block);
        const V = x.vocab;
        const chunks = (V + chunk_cols - 1) / chunk_cols;
        if (chunks > 64) return error.UnsupportedDFlash2Config;
        const widest = @max(@max(hidden, 2 * kv_width), @max(2 * inter, qkv_width));
        x.tmp = try x.zeroed(@max(write_rows * hidden, block * widest) * f32_bytes);
        x.c1 = try x.zeroed(rows * hidden * bf16_bytes);
        x.ctx = try x.zeroed(rows * hidden * bf16_bytes);
        x.ckv = try x.zeroed(rows * 2 * kv_width * bf16_bytes);
        x.x = try x.zeroed(block * hidden * bf16_bytes);
        x.y = try x.zeroed(block * hidden * bf16_bytes);
        x.y2 = try x.zeroed(block * hidden * bf16_bytes);
        x.dyn = try x.zeroed(block * dyn_width * bf16_bytes);
        x.qkv = try x.zeroed(block * qkv_width * bf16_bytes);
        x.q = try x.zeroed(block * q_width * bf16_bytes);
        x.kb = try x.zeroed(block * kv_width * bf16_bytes);
        x.vb = try x.zeroed(block * kv_width * bf16_bytes);
        x.att = try x.zeroed(block * q_width * bf16_bytes);
        x.a = try x.zeroed(block * hidden * bf16_bytes);
        x.gu = try x.zeroed(block * 2 * inter * bf16_bytes);
        x.act = try x.zeroed(block * inter * bf16_bytes);
        x.logits = try x.zeroed(proposals * V * f32_bytes);
        x.pv = try x.zeroed(proposals * chunks * top * 4);
        x.pi = try x.zeroed(proposals * chunks * top * 4);
        x.cand = try x.zeroed(proposals * 2 * top * 4);
        x.g = try x.zeroed(block * rank * f32_bytes);
        x.pm = try x.zeroed(block * q_heads * attn_chunks * 4);
        x.pl = try x.zeroed(block * q_heads * attn_chunks * 4);
        x.pacc = try x.zeroed(block * q_heads * attn_chunks * head_dim * 4);
        x.ids = try x.zeroed(block * 4);
        x.bpos = try x.zeroed(block * 4);
        x.ring_ptr = try x.zeroed(8);
        x.start = try x.zeroed(8);
        x.from = try x.zeroed(8);
        x.rings = try x.zeroed(@as(usize, x.streams) * ring_bytes);
        x.walks = try x.zeroed(@as(usize, x.streams) * walk_bytes);
        x.walk_table = try x.zeroed(@as(usize, x.streams) * 8);
        var table: [64]u64 = undefined;
        if (x.streams > table.len) return error.UnsupportedDFlash2Config;
        for (table[0..x.streams], 0..) |*t, i| t.* = x.walks.ptr + i * walk_bytes;
        try x.walk_table.upload(0, std.mem.sliceAsBytes(table[0..x.streams]));
    }

    pub fn deinit(x: *Dflash2) void {
        inline for (.{ "tmp", "c1", "ctx", "ckv", "x", "y", "y2", "dyn", "qkv", "q", "kb", "vb", "att", "a", "gu", "act", "logits", "pv", "pi", "cand", "g", "pm", "pl", "pacc", "ids", "bpos", "ring_ptr", "start", "from", "rings", "walks", "walk_table", "fc", "hproj", "hnorm", "norm", "pred", "succ" }) |f| @field(x, f).free();
        for (&x.layers) |*l| inline for (.{ "qkv", "kv", "o", "gu", "down", "a_proj", "m_proj", "in_norm", "post_norm", "q_norm", "k_norm", "a_base", "m_base" }) |f| @field(l, f).free();
        x.gpa.free(x.lane);
        x.k.mod.unload();
        x.* = undefined;
    }

    fn ringBase(x: *const Dflash2, stream: u32) u64 {
        return x.rings.ptr + @as(u64, stream) * ring_bytes;
    }

    /// out bf16 [m, n] = round(a bf16 [m, k] . w bf16 [n, k]^T).
    fn gemm(x: *Dflash2, s: cuda.Stream, a: u64, lda: usize, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        try x.c.dense.rows(s, a, lda, w, x.tmp.ptr, n, m, n, k);
        try ops.tobf(&x.c.ops, s, x.tmp.ptr, m, n, n, out, n, 0, n);
    }

    /// out bf16 [block, n] for the block pass: `w` as `putW` stored it.
    fn lin(x: *Dflash2, s: cuda.Stream, a: u64, w: u64, out: u64, n: usize, k: usize) !void {
        if (!x.q8) return x.gemm(s, a, k, w, out, block, n, k);
        const seg: ops.DenseSeg = .{ .x = a, .w = w, .out = out, .n = @intCast(n), .block0 = 0 };
        try ops.dense1Q8Rows(&x.c.ops, s, &.{seg}, k, false, 0, block);
    }

    fn normRows(x: *Dflash2, s: cuda.Stream, in: u64, w: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ in, w, out }) |v| a.add(v);
        a.add(eps);
        try go(x.k.norm, s, .{ .x = @intCast(rows) }, 256, &a);
    }

    /// `rows` committed positions from `pos0` of `stream`, their taps (bf16 [rows][taps][hidden]) at `tap_rows`: fc, the hidden
    /// norm, each layer's k (normed, rotated) and v into the stream's ring.
    pub fn writeContext(x: *Dflash2, s: cuda.Stream, stream: u32, tap_rows: u64, rows: usize, pos0: u64) !void {
        if (rows == 0 or rows > write_rows) return error.RowsExceedScratch;
        const l = &x.lane[stream];
        if (pos0 == 0 or pos0 > l.end) l.from = pos0;
        try x.gemm(s, tap_rows, taps.len * hidden, x.fc.ptr, x.c1.ptr, rows, hidden, taps.len * hidden);
        try x.normRows(s, x.c1.ptr, x.hnorm.ptr, x.ctx.ptr, rows);
        for (x.layers, 0..) |ly, li| {
            try x.gemm(s, x.ctx.ptr, hidden, ly.kv.ptr, x.ckv.ptr, rows, 2 * kv_width, hidden);
            var a: cuda.Args = .{};
            a.add(x.ckv.ptr);
            a.add(ly.k_norm.ptr);
            a.add(@as(i64, @intCast(pos0)));
            a.add(x.ringBase(stream));
            a.add(@as(i32, @intCast(li)));
            a.add(eps);
            a.add(theta);
            try go(x.k.ctx_kv, s, .{ .x = @intCast(rows), .y = kv_heads }, 64, &a);
        }
        l.end = pos0 + rows;
    }

    /// The ring of a prompt state restored at `at` positions: the last window of them.
    pub fn restored(x: *Dflash2, stream: u32, at: u64) void {
        x.lane[stream] = .{ .from = at - @min(at, window), .end = at };
    }

    pub fn setRnnContext(_: *Dflash2, _: cuda.Stream, _: u32, _: u64, _: usize) !void {}

    fn conv(x: *Dflash2, s: cuda.Stream, z: u64, which: i32, base: u64, out: u64, add: bool) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ z, x.dyn.ptr }) |v| a.add(v);
        a.add(which);
        for ([_]u64{ base, out }) |v| a.add(v);
        a.add(@as(i32, @intFromBool(add)));
        try go(x.k.conv, s, .{ .x = block, .y = (hidden + 1023) / 1024 }, 256, &a);
    }

    /// The `proposals` tokens after `anchor` (position `pos`, the pending token) into `out`, and each one's own probability.
    pub fn propose(x: *Dflash2, s: cuda.Stream, stream: u32, anchor: u32, pos: u64, out: *[proposals]u32, chain: *[proposals]f32) !void {
        const l = &x.lane[stream];
        if (l.end != pos) return error.DraftContextGap;
        var ids: [block]u32 = undefined;
        var ids_i: [block]i32 = undefined;
        var bpos: [block]i32 = undefined;
        for (0..block) |t| {
            ids[t] = if (t == 0) anchor else mask_token;
            ids_i[t] = @intCast(ids[t]);
            bpos[t] = @intCast(pos + t);
        }
        try x.ids.uploadAsync(0, std.mem.sliceAsBytes(&ids_i), s.handle);
        try x.bpos.uploadAsync(0, std.mem.sliceAsBytes(&bpos), s.handle);
        const ring: [1]u64 = .{x.ringBase(stream)};
        const start: [1]i64 = .{@intCast(pos)};
        const from: [1]i64 = .{@intCast(l.from)};
        try x.ring_ptr.uploadAsync(0, std.mem.sliceAsBytes(&ring), s.handle);
        try x.start.uploadAsync(0, std.mem.sliceAsBytes(&start), s.handle);
        try x.from.uploadAsync(0, std.mem.sliceAsBytes(&from), s.handle);
        try decode.embedTokens(x.c, s, &ids, x.x.ptr);
        for (x.layers, 0..) |ly, li| {
            try x.normRows(s, x.x.ptr, ly.in_norm.ptr, x.y.ptr, block);
            try x.lin(s, x.y.ptr, ly.a_proj.ptr, x.dyn.ptr, dyn_width, hidden);
            try x.conv(s, x.y.ptr, 0, ly.a_base.ptr, x.y2.ptr, false);
            try x.lin(s, x.y2.ptr, ly.qkv.ptr, x.qkv.ptr, qkv_width, hidden);
            {
                var a: cuda.Args = .{};
                a.add(x.qkv.ptr);
                a.add(@as(i32, qkv_width));
                for ([_]u64{ ly.q_norm.ptr, ly.k_norm.ptr, x.bpos.ptr }) |v| a.add(v);
                a.add(eps);
                a.add(theta);
                for ([_]u64{ x.q.ptr, x.kb.ptr, x.vb.ptr }) |v| a.add(v);
                try go(x.k.block_qkv, s, .{ .x = block, .y = q_heads + kv_heads }, 64, &a);
            }
            {
                var a: cuda.Args = .{};
                for ([_]u64{ x.q.ptr, x.kb.ptr, x.vb.ptr, x.ring_ptr.ptr, x.start.ptr, x.from.ptr }) |v| a.add(v);
                a.add(@as(i32, @intCast(li)));
                a.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
                for ([_]u64{ x.pm.ptr, x.pl.ptr, x.pacc.ptr }) |v| a.add(v);
                try go(x.k.attn_part, s, .{ .x = 1, .y = kv_heads, .z = attn_chunks }, 256, &a);
                var b: cuda.Args = .{};
                for ([_]u64{ x.pm.ptr, x.pl.ptr, x.pacc.ptr, x.att.ptr }) |v| b.add(v);
                try go(x.k.attn_merge, s, .{ .x = block, .y = q_heads }, head_dim, &b);
            }
            try x.lin(s, x.att.ptr, ly.o.ptr, x.a.ptr, hidden, q_width);
            try x.conv(s, x.a.ptr, 1, ly.a_base.ptr + 2 * hidden * bf16_bytes, x.x.ptr, true);
            try x.normRows(s, x.x.ptr, ly.post_norm.ptr, x.y.ptr, block);
            try x.lin(s, x.y.ptr, ly.m_proj.ptr, x.dyn.ptr, dyn_width, hidden);
            try x.conv(s, x.y.ptr, 0, ly.m_base.ptr, x.y2.ptr, false);
            try x.lin(s, x.y2.ptr, ly.gu.ptr, x.gu.ptr, 2 * inter, hidden);
            {
                var a: cuda.Args = .{};
                a.add(x.gu.ptr);
                a.add(x.act.ptr);
                a.add(@as(i32, inter));
                try go(x.k.swiglu, s, .{ .x = block, .y = (inter + 1023) / 1024 }, 256, &a);
            }
            try x.lin(s, x.act.ptr, ly.down.ptr, x.a.ptr, hidden, inter);
            try x.conv(s, x.a.ptr, 1, ly.m_base.ptr + 2 * hidden * bf16_bytes, x.x.ptr, true);
        }
        try x.normRows(s, x.x.ptr, x.norm.ptr, x.y.ptr, block);
        // rows 1.. through the target's head, their 16 best, the selector's projections, the greedy path
        const was = x.c.window;
        x.c.window = true;
        defer x.c.window = was;
        try head.logitsRaw(x.c, s, x.y.ptr + hidden * bf16_bytes, proposals, x.logits.ptr);
        const chunks = (x.vocab + chunk_cols - 1) / chunk_cols;
        {
            var a: cuda.Args = .{};
            a.add(x.logits.ptr);
            a.add(@as(i32, @intCast(x.vocab)));
            a.add(x.pv.ptr);
            a.add(x.pi.ptr);
            try go(x.k.top16_part, s, .{ .x = proposals, .y = @intCast(chunks) }, 256, &a);
            var b: cuda.Args = .{};
            b.add(x.pv.ptr);
            b.add(x.pi.ptr);
            b.add(@as(i32, @intCast(chunks)));
            b.add(x.cand.ptr);
            try go(x.k.top16_merge, s, .{ .x = proposals }, 256, &b);
        }
        try x.c.dense.rows(s, x.y.ptr, hidden, x.hproj.ptr, x.g.ptr, rank, block, rank, hidden);
        {
            var a: cuda.Args = .{};
            for ([_]u64{ x.ids.ptr, x.cand.ptr, x.g.ptr, x.pred.ptr, x.succ.ptr, x.walk_table.ptr + @as(u64, stream) * 8 }) |v| a.add(v);
            a.add(cap);
            try go(x.k.select, s, .{ .x = 1 }, 256, &a);
        }
        var got: [2 * block]u32 = undefined;
        try x.walks.downloadAsync(@as(usize, stream) * walk_bytes, std.mem.sliceAsBytes(&got), s.handle);
        try s.synchronize();
        for (out, got[1..block]) |*o, g| {
            if (g >= x.vocab) return error.Inkling2DraftToken;
            o.* = g;
        }
        for (chain, got[block + 1 ..]) |*cc, g| cc.* = @bitCast(g);
    }

    /// A stream's context at `at` positions, in image order: each layer's last `min(at, window)` K rows, then its V rows (position
    /// p at slot p % window). `f` gets each run's device address, its offset in the image and its length.
    pub fn each(x: *const Dflash2, stream: u32, at: u32, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: u64, len: u64) anyerror!void) !void {
        const row: u64 = kv_width * bf16_bytes;
        var off: u64 = 0;
        for (0..layers_n) |li| for ([_]u64{ 0, 1 }) |kv| {
            const base = x.ringBase(stream) + (@as(u64, li) * 2 + kv) * window * row;
            const keep = @min(at, window);
            var p: u64 = at - keep;
            while (p < at) {
                const slot = p % window;
                const run = @min(at - p, window - slot);
                try f(ctx, base + slot * row, off, run * row);
                off += run * row;
                p += run;
            }
        };
    }

    /// The bytes `each` copies for `at` positions.
    pub fn snapBytes(_: *const Dflash2, at: u32) u64 {
        return @as(u64, layers_n) * 2 * @min(at, window) * kv_width * bf16_bytes;
    }
};

/// Whether `dir` holds a DFlash2 draft model (a config.json with a dflash_config), not a target checkpoint.
pub fn isDflash2(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return false;
    defer gpa.free(text);
    const p = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return false;
    defer p.deinit();
    return p.value == .object and p.value.object.get("dflash_config") != null;
}

fn int(o: std.json.ObjectMap, key: []const u8) ?i64 {
    const v = o.get(key) orelse return null;
    return if (v == .integer) v.integer else null;
}

fn num(o: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn is(o: std.json.ObjectMap, key: []const u8, want: i64) bool {
    return (int(o, key) orelse return false) == want;
}

fn near(o: std.json.ObjectMap, key: []const u8, want: f32) bool {
    return @abs((num(o, key) orelse return false) - want) <= 1e-6 * @abs(@as(f64, want));
}

/// The draft's config.json must be the DFlash2 head this code was written for.
pub fn admit(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20)) catch return error.DFlash2MissingFile;
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer p.deinit();
    if (p.value != .object) return error.UnsupportedDFlash2Config;
    const o = p.value.object;
    const dcv = o.get("dflash_config") orelse return error.UnsupportedDFlash2Config;
    if (dcv != .object) return error.UnsupportedDFlash2Config;
    const dc = dcv.object;
    var ok = is(o, "hidden_size", hidden) and is(o, "num_hidden_layers", layers_n) and is(o, "num_attention_heads", q_heads) and
        is(o, "num_key_value_heads", kv_heads) and is(o, "head_dim", head_dim) and is(o, "intermediate_size", inter) and
        is(o, "sliding_window", window) and near(o, "rms_norm_eps", eps) and
        is(dc, "block_size", block) and is(dc, "mask_token_id", mask_token) and is(dc, "selector_top_k", top) and
        is(dc, "selector_rank", rank) and is(dc, "conv_kernel_size", 2) and is(dc, "conv_group_size", group) and
        near(dc, "final_logit_softcapping", cap);
    if (o.get("rope_parameters")) |rp| {
        if (rp != .object or !near(rp.object, "rope_theta", theta)) ok = false;
    }
    if (dc.get("selector_softcap_mode")) |m| {
        if (m != .string or !std.mem.eql(u8, m.string, "after_correction")) ok = false;
    }
    if (dc.get("target_layer_ids")) |tv| {
        if (tv != .array or tv.array.items.len != taps.len) {
            ok = false;
        } else for (tv.array.items, taps) |t, want| {
            if (t != .integer or t.integer != want) ok = false;
        }
    } else ok = false;
    if (!ok) return error.UnsupportedDFlash2Config;
}

test "a ring holds a window of every layer's K and V and a tapped row is four hidden rows" {
    try std.testing.expectEqual(@as(usize, 5 * 2 * 2048 * 1024 * 2), ring_bytes);
    try std.testing.expectEqual(@as(usize, 576), dyn_width);
}
