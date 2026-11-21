//! Decode forward of Inkling 2 Mini for the visits before the loop restarts (ref.model.Model.forward, _visit and
//! _next_input at one row): embedding, attention, the two sublayer convs, the routed MoE and the residual mixes.
const std = @import("std");
const cuda = @import("cuda");
const attn_visit = @import("attn_visit.zig");
const bank = @import("bank.zig");
const config = @import("config.zig");
const engine = @import("engine.zig");
const loop_kernels = @import("loop_kernels.zig");
const moe = @import("moe.zig");
const ops = @import("ops.zig");
const visits = @import("visits.zig");
const weights = @import("weights.zig");

const hidden = moe.hidden;
const rows = 1;
const bf16_bytes = 2;
const f32_bytes = 4;
/// The attention input of a global layer has q, k and v rows; a local layer's has one.
const global_inputs = engine.max_inputs;
const mlp_queries = 1;

/// Called after visit `v` with the carried float32 [qn, hidden] vector the next visit reads (SPEC 5 step E), and by
/// `mlp` with the visit's own float32 [hidden] MoE input before a teacher replaces it.
pub const Trace = struct {
    ctx: *anyopaque,
    visit: *const fn (ctx: *anyopaque, v: u32, inp: u64, qn: u32) anyerror!void,
    mlp: ?*const fn (ctx: *anyopaque, v: u32, mlp_in: u64) anyerror!void = null,
};

/// Gives the carried input of visit `v`, and by `mlp` its MoE input, as the reference computed them; the engine reads
/// them in place of its own.
pub const Teacher = struct {
    ctx: *anyopaque,
    input: *const fn (ctx: *anyopaque, v: u32) ?[]const f32,
    mlp: ?*const fn (ctx: *anyopaque, v: u32) ?[]const f32 = null,
};

pub const Hooks = struct { trace: ?Trace = null, teacher: ?Teacher = null };

/// Rows of the carried vector after visit `i`: q, k and v for a global next layer, else one.
pub fn carriedRows(sched: *const visits.Schedule, i: u32) u32 {
    if (i + 1 >= sched.n) return 1;
    return if (config.isGlobal(sched.visits[i + 1].layer)) global_inputs else 1;
}

/// The last visit this forward runs: the prelude and the first pass of the core. Later visits need the restart and messages.
pub fn lastVisit(loop: config.Loop) u32 {
    return loop.core_end - 1;
}

/// bf16 bits of `x`, round to nearest even.
pub fn bf16Bits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @truncate((u +% 0x7fff +% lsb) >> 16);
}

fn upload(c: *const engine.Ctx, s: cuda.Stream, addr: u64, bytes: []const u8) !void {
    const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = addr, .len = bytes.len, .borrowed = true };
    try view.uploadAsync(0, bytes, s.handle);
}

fn present(d: weights.Dev) !u64 {
    if (d.ptr == 0) return error.Inkling2MissingWeight;
    return d.ptr;
}

fn layerWeights(c: *const engine.Ctx, layer: u32) !*const weights.LayerW {
    if (layer < c.w.first or layer > c.w.last) return error.Inkling2LayerNotLoaded;
    return &c.w.layers[layer];
}

/// The norm weights of the attention input of `layer`: q, k, v for a global layer, the one norm for a local layer.
fn inputNorms(lw: *const weights.LayerW, layer: u32) ![ops.max_queries]u64 {
    if (!config.isGlobal(layer)) return .{ try present(lw.attn_norm orelse return error.Inkling2MissingWeight), 0, 0 };
    return .{
        try present(lw.q_norm orelse return error.Inkling2MissingWeight),
        try present(lw.k_norm orelse return error.Inkling2MissingWeight),
        try present(lw.v_norm orelse return error.Inkling2MissingWeight),
    };
}

/// Row `token` of the embedding through embed_norm into `out`; a host-resident table goes up as one scratch row.
fn embedToken(c: *engine.Ctx, s: cuda.Stream, token: u32, out: u64) !void {
    const w = c.w;
    const f = c.flow.r;
    var table: u64 = undefined;
    var id: i32 = undefined;
    if (w.embed_host != null) {
        var row: [hidden]u16 = undefined;
        try w.gatherEmbed(&.{token}, &row);
        try upload(c, s, f.emb, std.mem.sliceAsBytes(&row));
        table = f.emb;
        id = 0;
    } else {
        table = try present(w.embed orelse return error.Inkling2MissingWeight);
        id = std.math.cast(i32, token) orelse return error.TokenOutOfRange;
    }
    try upload(c, s, f.ids, std.mem.asBytes(&id));
    try ops.embed(&c.ops, s, f.ids, rows, table, try present(w.embed_norm), out, hidden, c.text.eps);
}

fn mixArgs(c: *const engine.Ctx, blocks: []const u64, partial: u64, partial_bf16: bool, query: u64, norms: [ops.max_queries]u64, nq: u32, out: u64) ops.MixArgs {
    var m: ops.MixArgs = .{
        .partial = partial,
        .query = query,
        .norm = norms,
        .out = out,
        .n_blocks = @intCast(blocks.len),
        .partial_bf16 = @intFromBool(partial_bf16),
        .nq = @intCast(nq),
        .rows = rows,
        .c = hidden,
        .eps = c.text.eps,
    };
    @memcpy(m.blocks[0..blocks.len], blocks);
    return m;
}

fn conv(c: *engine.Ctx, s: cuda.Stream, a: ops.ConvArgs) !void {
    try ops.conv(&c.ops, s, a, 1);
}

/// Visit `i`'s MoE input to the trace, then the teacher's in its place: the activation quantiser turns a sub-ulp input
/// difference into a percent-level output difference, which the stack and the MLP conv state would carry on.
fn moeInput(c: *engine.Ctx, s: cuda.Stream, i: u32, h: Hooks) !void {
    const f = c.flow.r;
    if (h.trace) |tr| if (tr.mlp) |put| {
        try loop_kernels.mixAdd(&c.loop, s, f.wide, f.mlp_in, 0, rows * hidden);
        try s.synchronize();
        try put(tr.ctx, i, f.wide);
    };
    const te = h.teacher orelse return;
    const get = te.mlp orelse return;
    const vec = get(te.ctx, i) orelse return;
    if (vec.len != rows * hidden) return error.Inkling2TeacherShape;
    var bits: [rows * hidden]u16 = undefined;
    for (vec, &bits) |x, *b| b.* = bf16Bits(x);
    try upload(c, s, f.mlp_in, std.mem.sliceAsBytes(&bits));
    try s.synchronize(); // the host copy goes out of scope
}

/// Steps A-D of ref `_visit`: attention, the attention conv, the MLP input mix, the MoE, the MLP conv and the residual update.
fn sublayers(c: *engine.Ctx, s: cuda.Stream, stream: u32, pos: u64, i: u32, v: visits.Visit, op: bank.Op, h: Hooks) !void {
    const f = c.flow.r;
    const lw = try layerWeights(c, v.layer);
    const row_bytes = rows * hidden * bf16_bytes;
    const x: [3]u64 = if (config.isGlobal(v.layer)) .{ f.inp, f.inp + row_bytes, f.inp + 2 * row_bytes } else .{ f.inp, f.inp, f.inp };
    try attn_visit.run(c, s, v, lw, x, rows, pos, stream, f.attn_pre);
    try conv(c, s, .{ .x = f.attn_pre, .state = c.cache.convAt(v.conv, stream, .attn), .w = try present(lw.attn_sconv), .out = f.attn_out, .c = hidden, .rows = rows, .mode = 0, .x_bf16 = 1 });

    const query = try present(c.w.attn_res.mlp[v.layer]);
    const norm: [ops.max_queries]u64 = .{ try present(lw.mlp_norm), 0, 0 };
    if (op.first) {
        try ops.mix(&c.ops, s, mixArgs(c, f.stack[0 .. op.blocks + 1], f.attn_out, true, query, norm, mlp_queries, f.mlp_in));
    } else {
        try loop_kernels.mixAdd(&c.loop, s, f.resid, f.stack[op.blocks], f.attn_out, rows * hidden);
        try ops.mix(&c.ops, s, mixArgs(c, f.stack[0..op.blocks], f.resid, false, query, norm, mlp_queries, f.mlp_in));
    }
    try moeInput(c, s, i, h);
    try moe.forward(&c.env, s, lw, c.w.prec[v.layer], f.mlp_in, f.moe_out, rows, &c.moe_sc);

    var a: ops.ConvArgs = .{ .x = f.moe_out, .state = c.cache.convAt(v.conv, stream, .mlp), .w = try present(lw.mlp_sconv), .resid = f.resid, .c = hidden, .rows = rows, .mode = 1, .x_bf16 = 1, .y_f32 = 1 };
    if (op.first) {
        a.base = f.attn_out;
        try conv(c, s, a);
        try ops.tobf(&c.ops, s, f.resid, rows, hidden, hidden, f.stack[op.blocks + 1], hidden, 0, hidden);
    } else {
        a.mode = 2;
        a.out = f.stack[op.blocks];
        try conv(c, s, a);
    }
}

/// Step E: the vector the next visit reads, into `inp` (bf16, normed) or, at the restart, as the saved `dest` (fp32).
fn nextInput(c: *engine.Ctx, s: cuda.Stream, i: u32, v: visits.Visit, op: bank.Op) !void {
    const f = c.flow.r;
    const blocks = op.blocks + @intFromBool(op.first);
    switch (op.next) {
        .mix_norm => {
            const nv = c.sched.visits[i + 1];
            const nlw = try layerWeights(c, nv.layer);
            const nq = carriedRows(&c.sched, i);
            const query = try present(c.w.attn_res.attn[v.layer]);
            const norms = try inputNorms(nlw, nv.layer);
            try ops.mix(&c.ops, s, mixArgs(c, f.stack[0..blocks], f.stack[blocks], true, query, norms, nq, f.inp));
            if (op.save_dests) {
                if (nq != 1) return error.Inkling2DestRows;
                try loop_kernels.mixF32(&c.loop, s, mixArgs(c, f.stack[0..blocks], f.stack[blocks], true, query, .{ 0, 0, 0 }, nq, f.dest));
            }
        },
        .prelude_dests => {},
        .mix_fp32, .head => return error.Inkling2LoopVisit,
    }
}

/// What the trace sees after a visit: the carried vector as float32 and its row count.
fn traced(c: *engine.Ctx, s: cuda.Stream, i: u32, op: bank.Op) !struct { ptr: u64, qn: u32 } {
    const f = c.flow.r;
    if (op.next == .prelude_dests) return .{ .ptr = f.dest, .qn = 1 };
    const qn = carriedRows(&c.sched, i);
    try loop_kernels.mixAdd(&c.loop, s, f.wide, f.inp, 0, qn * rows * hidden);
    return .{ .ptr = f.wide, .qn = qn };
}

/// Replaces the carried input before visit `i + 1` with the teacher's vector (`qn` rows of float32).
fn teach(c: *engine.Ctx, s: cuda.Stream, op: bank.Op, qn: u32, vec: []const f32) !void {
    const f = c.flow.r;
    if (vec.len != qn * rows * hidden) return error.Inkling2TeacherShape;
    if (op.next == .prelude_dests) {
        try upload(c, s, f.dest, std.mem.sliceAsBytes(vec));
    } else {
        var bits: [global_inputs * hidden]u16 = undefined;
        for (vec, bits[0..vec.len]) |x, *b| b.* = bf16Bits(x);
        try upload(c, s, f.inp, std.mem.sliceAsBytes(bits[0..vec.len]));
    }
    try s.synchronize(); // the host copy goes out of scope
}

/// Visits `0..last_visit` of the schedule for one token at position `pos` of `stream`. The carried vector after each
/// visit and each visit's MoE input go to `h.trace`; `h.teacher` replaces both, so errors cannot accumulate.
pub fn step(c: *engine.Ctx, s: cuda.Stream, stream: u32, token: u32, pos: u64, last_visit: u32, h: Hooks) !void {
    const loop = c.w.cfg.loop orelse return error.NotLoop;
    if (last_visit > lastVisit(loop) or last_visit >= c.sched.n) return error.Inkling2LoopVisit;
    if (c.text.hidden != hidden) return error.Inkling2Hidden;
    const plan = bank.plan(&c.sched, &c.w.cfg);
    const f = c.flow.r;

    try embedToken(c, s, token, f.stack[0]);
    const first_norms = try inputNorms(try layerWeights(c, c.sched.visits[0].layer), c.sched.visits[0].layer);
    const nq: u32 = if (config.isGlobal(c.sched.visits[0].layer)) global_inputs else 1;
    try ops.mix(&c.ops, s, mixArgs(c, &.{}, f.stack[0], true, 0, first_norms, nq, f.inp));

    for (0..last_visit + 1) |idx| {
        const i: u32 = @intCast(idx);
        const v = c.sched.visits[i];
        const op = plan[i];
        try sublayers(c, s, stream, pos, i, v, op, h);
        try nextInput(c, s, i, v, op);
        if (i == last_visit and h.trace == null) break;
        if (h.trace == null and h.teacher == null) continue;
        const t = try traced(c, s, i, op);
        if (h.trace) |tr| {
            try s.synchronize();
            try tr.visit(tr.ctx, i, t.ptr, t.qn);
        }
        if (i < last_visit) if (h.teacher) |te| if (te.input(te.ctx, i)) |vec| try teach(c, s, op, t.qn, vec);
    }
    try s.synchronize();
}

const ts = std.testing;
const plan_mod = @import("plan.zig");
const config_test = @import("config_test.zig");

fn miniSchedule() !struct { cfg: config.Config, s: visits.Schedule } {
    var why: config.Why = .{};
    const cfg = try config.parse(ts.allocator, config_test.mini, &why);
    const prec: [config.max_layers]plan_mod.Precision = @splat(.nvfp4);
    const p = try plan_mod.build(&cfg.text, prec[0..cfg.text.layers]);
    return .{ .cfg = cfg, .s = try visits.build(&cfg, &p) };
}

test "carried rows follow the next layer's kind" {
    const m = try miniSchedule();
    for (0..m.s.n) |i| {
        const next_global = i + 1 < m.s.n and config.isGlobal(m.s.visits[i + 1].layer);
        try ts.expectEqual(@as(u32, if (next_global) 3 else 1), carriedRows(&m.s, @intCast(i)));
    }
    try ts.expectEqual(@as(u32, 3), carriedRows(&m.s, 4));
    try ts.expectEqual(@as(u32, 1), carriedRows(&m.s, 5)); // layer 6 is local: the prelude destinations have one row
    try ts.expectEqual(@as(u32, 1), carriedRows(&m.s, 17));
    try ts.expectEqual(@as(u32, 1), carriedRows(&m.s, 47));
}

test "the forward stops where the loop restarts" {
    const m = try miniSchedule();
    try ts.expectEqual(@as(u32, 17), lastVisit(m.cfg.loop.?));
    try ts.expect(m.s.visits[18].restart);
}

test "bf16 rounding is nearest even and keeps widened bf16 values" {
    try ts.expectEqual(@as(u16, 0x3f80), bf16Bits(1.0));
    try ts.expectEqual(@as(u16, 0x3f80), bf16Bits(@bitCast(@as(u32, 0x3f807fff)))); // below the tie
    try ts.expectEqual(@as(u16, 0x3f80), bf16Bits(@bitCast(@as(u32, 0x3f808000)))); // tie to even
    try ts.expectEqual(@as(u16, 0x3f82), bf16Bits(@bitCast(@as(u32, 0x3f818000)))); // tie to even, up
    try ts.expectEqual(@as(u16, 0x3f81), bf16Bits(@bitCast(@as(u32, 0x3f808001)))); // above the tie
    for ([_]u16{ 0x0000, 0x8000, 0x3c23, 0xc0a5, 0x7f7f }) |b| try ts.expectEqual(b, bf16Bits(@bitCast(@as(u32, b) << 16)));
    const nan = bf16Bits(std.math.nan(f32));
    try ts.expect(std.math.isNan(@as(f32, @bitCast(@as(u32, nan) << 16))));
}

test "step refuses visits past the first pass before touching the device" {
    var c: engine.Ctx = undefined;
    var w: weights.Weights = undefined;
    const m = try miniSchedule();
    w.cfg = m.cfg;
    c.w = &w;
    c.sched = m.s;
    var d: cuda.Driver = undefined;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try ts.expectError(error.Inkling2LoopVisit, step(&c, s, 0, 0, 0, 18, .{}));
    try ts.expectError(error.Inkling2LoopVisit, step(&c, s, 0, 0, 0, 47, .{}));
}
