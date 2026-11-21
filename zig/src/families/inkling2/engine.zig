//! Device context of Inkling 2 Mini's engine: the kernel images, the visit caches and the attention scratch.
const std = @import("std");
const cuda = @import("cuda");
const bank = @import("bank.zig");
const cache_mod = @import("cache.zig");
const config = @import("config.zig");
const kernels = @import("kernels.zig");
const loop_kernels = @import("loop_kernels.zig");
const moe = @import("moe.zig");
const ops = @import("ops.zig");
const visits = @import("visits.zig");
const weights = @import("weights.zig");

pub const Cache = cache_mod.Cache;

/// Rows of a call the metadata arrays are sized for.
pub const max_chunk_rows = 4096;
const region_align = 256;

/// The attention shapes a layer's scratch depends on; Mini's local and global layers share them.
pub const Geometry = struct {
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    stride: u32, // partials a (row, head): the most chunks any layer's keys span
    max_rows: u32,

    pub fn of(t: *const config.Text, o: cache_mod.Options) !Geometry {
        if (!std.meta.eql(t.local, t.global)) return error.Inkling2MixedHeads;
        if (t.local.dim != kernels.head_dim or t.d_rel != kernels.d_rel or t.rel_extent != kernels.extent or t.window != kernels.extent) return error.Inkling2AttentionShape;
        if (o.chunk_rows > max_chunk_rows) return error.RowsExceedScratch;
        const sliding = kernels.slotsFor(kernels.Layer.sliding(t.window, o.chunk_rows), o.context);
        const global = kernels.slotsFor(kernels.Layer.global(@intCast(o.context)), o.context);
        return .{ .hidden = t.hidden, .heads = t.local.q, .kv_heads = t.local.kv, .dim = t.local.dim, .stride = @max(sliding, global), .max_rows = o.chunk_rows };
    }

    pub fn qDim(g: Geometry) usize {
        return @as(usize, g.heads) * g.dim;
    }

    pub fn kvDim(g: Geometry) usize {
        return @as(usize, g.kv_heads) * g.dim;
    }

    pub fn relDim(g: Geometry) usize {
        return @as(usize, g.heads) * kernels.d_rel;
    }

    pub fn group(g: Geometry) u32 {
        return g.heads / g.kv_heads;
    }

    /// Partial slots: every row and head at `stride` chunks, or a split call's sub-chunks for at most `split_rows` rows.
    pub fn slots(g: Geometry) usize {
        const whole = @as(usize, g.max_rows) * g.heads * g.stride;
        const split = @as(usize, @min(g.max_rows, kernels.split_rows)) * g.heads * g.stride * kernels.subs;
        return @max(whole, split);
    }
};

/// Addresses of the attention scratch; `of(0, g)` sizes it.
pub const Regions = struct {
    tmp: u64, // fp32 [rows, max(q, hidden)]: a projection before its bf16 rounding
    q_raw: u64,
    q: u64,
    gate: u64,
    out: u64, // the merged attention rows, bf16 [rows, q]
    r_raw: u64,
    rel: u64,
    sink: u64,
    k_raw: u64,
    v_raw: u64,
    k_conv: u64,
    v_conv: u64,
    dyn: u64, // fp32 [rows, kv_heads * conv_taps]: dynamic conv taps at bf16 values
    table: u64,
    part_o: u64,
    part_ml: u64,
    seg_of: u64, // int32 [rows], all zero: one stream a call
    pos: u64,
    segs: u64,
    tiles: u64,
    total: u64,

    pub fn of(base: u64, g: Geometry) Regions {
        var at: u64 = base;
        var r: Regions = undefined;
        const rows: usize = g.max_rows;
        const wide = @max(g.qDim(), g.hidden);
        const sizes = .{
            .{ "tmp", rows * wide * 4 },
            .{ "q_raw", rows * g.qDim() * 2 },
            .{ "q", rows * g.qDim() * 2 },
            .{ "gate", rows * g.qDim() * 2 },
            .{ "out", rows * g.qDim() * 2 },
            .{ "r_raw", rows * g.relDim() * 2 },
            .{ "rel", rows * g.relDim() * 2 },
            .{ "sink", rows * g.heads * 2 },
            .{ "k_raw", rows * g.kvDim() * 2 },
            .{ "v_raw", rows * g.kvDim() * 2 },
            .{ "k_conv", rows * g.kvDim() * 2 },
            .{ "v_conv", rows * g.kvDim() * 2 },
            .{ "dyn", rows * g.kv_heads * config.conv_taps * 4 },
            .{ "table", kernels.tableBytes(rows, g.heads) },
            .{ "part_o", g.slots() * kernels.head_dim * 4 },
            .{ "part_ml", g.slots() * 8 },
            .{ "seg_of", rows * 4 },
            .{ "pos", rows * 4 },
            .{ "segs", @sizeOf(kernels.Seg) },
            .{ "tiles", kernels.tileCount(rows, g.group()) * @sizeOf(kernels.Tile) },
        };
        inline for (sizes) |s| {
            @field(r, s[0]) = at;
            at += std.mem.alignForward(u64, s[1], region_align);
        }
        r.total = at - base;
        return r;
    }
};

/// The attention scratch for calls of up to `geometry.max_rows` rows; one call at a time.
pub const Scratch = struct {
    geometry: Geometry,
    mem: cuda.DeviceBuffer,
    r: Regions,

    pub fn init(d: *const cuda.Driver, g: Geometry) !Scratch {
        var mem = try cuda.DeviceBuffer.alloc(d, Regions.of(0, g).total);
        errdefer mem.free();
        const r = Regions.of(mem.ptr, g);
        try cache_mod.zeroSync(d, r.seg_of, @as(usize, g.max_rows) * 4);
        return .{ .geometry = g, .mem = mem, .r = r };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.mem.free();
        sc.* = undefined;
    }
};

const bf16_bytes = 2;
const f32_bytes = 4;
const embed_ids_bytes = 4;
/// Queries of a global layer's attention input: q, k and v.
pub const max_inputs = 3;

/// Addresses of the decode flow buffers: the residual stack and message bank, the carried input and the visit's
/// intermediates. `of(0, rows, hidden)` sizes them.
pub const FlowRegions = struct {
    stack: [bank.stack_slots]u64, // bf16 [rows, hidden]: the embedding, three finished blocks and the partial
    bank: [bank.bank_slots]u64, // bf16 [rows, hidden]: the two blocks a restart saves
    dest: u64, // fp32 [rows, hidden]: the prelude destinations
    inp: u64, // bf16 [3, rows, hidden]: the carried attention input
    attn_pre: u64, // bf16 [rows, hidden]
    attn_out: u64,
    mlp_in: u64,
    moe_out: u64,
    resid: u64, // fp32 [rows, hidden]: partial plus attention output, then the new partial before its rounding
    wide: u64, // fp32 [3, rows, hidden]: the carried input widened for a trace
    ids: u64, // int32 [rows]
    emb: u64, // bf16 [rows, hidden]: the embedding rows of a host-resident table
    total: u64,

    pub fn of(base: u64, rows: usize, hidden: usize) FlowRegions {
        var at = base;
        var r: FlowRegions = undefined;
        const row = rows * hidden;
        for (&r.stack) |*p| p.* = take(&at, row * bf16_bytes);
        for (&r.bank) |*p| p.* = take(&at, row * bf16_bytes);
        r.dest = take(&at, row * f32_bytes);
        r.inp = take(&at, max_inputs * row * bf16_bytes);
        r.attn_pre = take(&at, row * bf16_bytes);
        r.attn_out = take(&at, row * bf16_bytes);
        r.mlp_in = take(&at, row * bf16_bytes);
        r.moe_out = take(&at, row * bf16_bytes);
        r.resid = take(&at, row * f32_bytes);
        r.wide = take(&at, max_inputs * row * f32_bytes);
        r.ids = take(&at, rows * embed_ids_bytes);
        r.emb = take(&at, row * bf16_bytes);
        r.total = at - base;
        return r;
    }

    fn take(at: *u64, len: usize) u64 {
        const p = at.*;
        at.* += std.mem.alignForward(u64, len, region_align);
        return p;
    }
};

/// The decode flow buffers for calls of up to `max_rows` rows; one call at a time.
pub const Flow = struct {
    mem: cuda.DeviceBuffer,
    r: FlowRegions,

    pub fn init(d: *const cuda.Driver, rows: usize, hidden: usize) !Flow {
        const mem = try cuda.DeviceBuffer.alloc(d, FlowRegions.of(0, rows, hidden).total);
        return .{ .mem = mem, .r = FlowRegions.of(mem.ptr, rows, hidden) };
    }

    pub fn deinit(f: *Flow) void {
        f.mem.free();
        f.* = undefined;
    }
};

/// What the engine's visits run against. Initialised in place: `env` points into it, so it never moves.
pub const Ctx = struct {
    d: *const cuda.Driver,
    env: moe.Env,
    loop: loop_kernels.Kernels,
    ops: ops.Kernels,
    attn: kernels.Kernels,
    dense: cuda.dense_rows.DenseRows,
    text: config.Text,
    sched: visits.Schedule,
    w: *const weights.Weights,
    cache: *Cache,
    scratch: Scratch,
    flow: Flow,
    moe_sc: moe.Scratch,

    pub fn init(c: *Ctx, d: *const cuda.Driver, w: *const weights.Weights, sched: visits.Schedule, cache: *Cache, o: cache_mod.Options) !void {
        const g = try Geometry.of(&w.cfg.text, o);
        c.d = d;
        c.text = w.cfg.text;
        c.sched = sched;
        c.w = w;
        c.cache = cache;
        c.loop = try loop_kernels.Kernels.load(d);
        errdefer c.loop.unload();
        c.ops = try ops.Kernels.load(d);
        errdefer c.ops.unload();
        c.attn = try kernels.Kernels.load(d);
        errdefer c.attn.unload();
        c.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer c.dense.unload();
        c.scratch = try Scratch.init(d, g);
        errdefer c.scratch.deinit();
        c.flow = try Flow.init(d, g.max_rows, g.hidden);
        errdefer c.flow.deinit();
        c.moe_sc = try moe.Scratch.init(d, g.max_rows);
        errdefer c.moe_sc.deinit();
        c.env = .{ .driver = d, .loop = &c.loop, .ops = &c.ops };
    }

    pub fn deinit(c: *Ctx) void {
        c.moe_sc.deinit();
        c.flow.deinit();
        c.scratch.deinit();
        c.dense.unload();
        c.attn.unload();
        c.ops.unload();
        c.loop.unload();
        c.* = undefined;
    }
};

const ts = std.testing;
const config_test = @import("config_test.zig");

fn miniText() !config.Config {
    var why: config.Why = .{};
    return config.parse(ts.allocator, config_test.mini, &why);
}

test "geometry of Mini" {
    const cfg = try miniText();
    const g = try Geometry.of(&cfg.text, .{ .context = 65536, .streams = 2, .chunk_rows = 512 });
    try ts.expectEqual(@as(u32, 24), g.heads);
    try ts.expectEqual(@as(u32, 3), g.group());
    try ts.expectEqual(@as(usize, 3072), g.qDim());
    try ts.expectEqual(@as(usize, 1024), g.kvDim());
    try ts.expectEqual(@as(usize, 384), g.relDim());
    try ts.expectEqual(@as(u32, 5), g.stride); // the sliding window's five 256-key chunks outnumber the four global ones
    const one = try Geometry.of(&cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 1 });
    try ts.expectEqual(@as(u32, 5), one.stride);
}

test "geometry refuses shapes the kernels do not take" {
    var cfg = try miniText();
    try ts.expectError(error.RowsExceedScratch, Geometry.of(&cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = max_chunk_rows + 1 }));
    cfg.text.global.q = 48;
    try ts.expectError(error.Inkling2MixedHeads, Geometry.of(&cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 1 }));
    cfg = try miniText();
    cfg.text.d_rel = 8;
    try ts.expectError(error.Inkling2AttentionShape, Geometry.of(&cfg.text, .{ .context = 2048, .streams = 1, .chunk_rows = 1 }));
}

test "scratch regions are aligned, disjoint and in call order" {
    const cfg = try miniText();
    const g = try Geometry.of(&cfg.text, .{ .context = 65536, .streams = 2, .chunk_rows = 64 });
    const base: u64 = 1 << 30;
    const r = Regions.of(base, g);
    const ordered = [_]u64{ r.tmp, r.q_raw, r.q, r.gate, r.out, r.r_raw, r.rel, r.sink, r.k_raw, r.v_raw, r.k_conv, r.v_conv, r.dyn, r.table, r.part_o, r.part_ml, r.seg_of, r.pos, r.segs, r.tiles };
    var prev: u64 = base;
    for (ordered) |at| {
        try ts.expect(at % region_align == 0);
        try ts.expect(at >= prev);
        prev = at;
    }
    try ts.expectEqual(r.total, Regions.of(0, g).total);
    try ts.expect(r.total >= kernels.tableBytes(64, 24) + g.slots() * kernels.head_dim * 4);
    // a 64-row call's global layer splits every chunk into sub-chunks
    try ts.expectEqual(@as(usize, 64 * 24 * 5 * kernels.subs), g.slots());
    try ts.expect(r.part_o + g.slots() * kernels.head_dim * 4 <= r.part_ml);
}

test "scratch for a 512-row chunk stays small against the budget slack" {
    const cfg = try miniText();
    const g = try Geometry.of(&cfg.text, .{ .context = 65536, .streams = 2, .chunk_rows = 512 });
    try ts.expect(Regions.of(0, g).total < 256 << 20);
}

test "flow buffers for one row and for a 512-row chunk" {
    const one = FlowRegions.of(0, 1, 2304);
    try ts.expectEqual(@as(u64, 4608), one.stack[1] - one.stack[0]); // a bf16 row
    try ts.expectEqual(@as(u64, 5 * 4608), one.bank[0]); // five stack slots
    try ts.expectEqual(@as(u64, 7 * 4608), one.dest); // then two bank slots
    try ts.expectEqual(@as(u64, 9216), one.inp - one.dest); // one fp32 dest row
    try ts.expectEqual(@as(u64, 3 * 4608), one.attn_pre - one.inp);
    try ts.expectEqual(@as(u64, 115456), one.total);
    const big = FlowRegions.of(0, 512, 2304);
    try ts.expectEqual(@as(u64, 512 * 4608), big.stack[1] - big.stack[0]);
    try ts.expectEqual(@as(u64, 7 * 512 * 4608), big.dest);
    try ts.expectEqual(@as(u64, 512 * 9216), big.inp - big.dest);
    try ts.expectEqual(@as(u64, 58_984_448), big.total);
}

test "flow regions are aligned and disjoint" {
    const r = FlowRegions.of(1 << 20, 64, 2304);
    const ordered = r.stack ++ r.bank ++ [_]u64{ r.dest, r.inp, r.attn_pre, r.attn_out, r.mlp_in, r.moe_out, r.resid, r.wide, r.ids, r.emb };
    var prev: u64 = 1 << 20;
    for (ordered) |at| {
        try ts.expect(at % region_align == 0 and at >= prev);
        prev = at;
    }
    try ts.expect(r.emb + 64 * 2304 * 2 <= (1 << 20) + r.total);
}
