//! Decode forward of Inkling 2 Mini at one row (ref.model.Model.forward, _visit and _next_input): embedding,
//! attention, the two sublayer convs, the routed MoE and the residual mixes, and for the loop the message bank, the
//! prelude restart, the cross-pass messages and the controllers (SPEC 9).
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
const bf16_row = rows * hidden * bf16_bytes;
const f32_row = rows * hidden * f32_bytes;
/// The attention input of a global layer has q, k and v rows; a local layer's has one.
const global_inputs = engine.max_inputs;
const mlp_queries = 1;
const head_queries = 1;
/// Controller streams in `loop_kernels.stream_names` order: the three attention inputs, then the MLP input.
const attn_streams = 3;
const mlp_stream = 3;
const no_norms: [ops.max_queries]u64 = @splat(0);

comptime {
    std.debug.assert(std.mem.eql(u8, loop_kernels.stream_names[mlp_stream], "mlp"));
    std.debug.assert(loop_kernels.n_streams == attn_streams + 1 and attn_streams == global_inputs);
}

/// Called after visit `v` with the carried float32 [qn, hidden] vector the next visit reads (SPEC 5 step E), by `mlp`
/// with the visit's own float32 [hidden] MoE input before a teacher replaces it, and for a visit with pass > 0 by `xs`
/// with its controller outputs xq, xk, xv (float32 [3, hidden]) and by `msg` with its messages (float32 [4, hidden]).
pub const Trace = struct {
    ctx: *anyopaque,
    visit: *const fn (ctx: *anyopaque, v: u32, inp: u64, qn: u32) anyerror!void,
    mlp: ?*const fn (ctx: *anyopaque, v: u32, mlp_in: u64) anyerror!void = null,
    xs: ?*const fn (ctx: *anyopaque, v: u32, xs: u64) anyerror!void = null,
    msg: ?*const fn (ctx: *anyopaque, v: u32, msg: u64) anyerror!void = null,
};

/// Gives the carried input of visit `v`, by `mlp` its MoE input and by `msg` a pass > 0 visit's messages (float32
/// [4, hidden]), as the reference computed them; the engine reads them in place of its own. A carried input before a
/// pass > 0 visit is the float32 `dest` mix, else bf16 values.
pub const Teacher = struct {
    ctx: *anyopaque,
    input: *const fn (ctx: *anyopaque, v: u32) ?[]const f32,
    mlp: ?*const fn (ctx: *anyopaque, v: u32) ?[]const f32 = null,
    msg: ?*const fn (ctx: *anyopaque, v: u32) ?[]const f32 = null,
};

pub const Hooks = struct { trace: ?Trace = null, teacher: ?Teacher = null };

/// Rows of the carried vector after visit `i`: q, k and v for a global next layer, else one.
pub fn carriedRows(sched: *const visits.Schedule, i: u32) u32 {
    if (i + 1 >= sched.n) return 1;
    return if (config.isGlobal(sched.visits[i + 1].layer)) global_inputs else 1;
}

/// The highest layer whose weights visits 0..`last_visit` read: their own and the next visit's input norms.
pub fn layersRead(sched: *const visits.Schedule, last_visit: u32) u32 {
    const end = @min(last_visit + 2, sched.n);
    var top: u32 = 0;
    for (sched.visits[0..end]) |v| top = @max(top, v.layer);
    return top;
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

fn controllers(lw: *const weights.LayerW) !*const [loop_kernels.n_streams]weights.Controller {
    if (lw.controllers) |*ctl| return ctl;
    return error.Inkling2MissingWeight;
}

/// Stream `stream` of a pass > 0 visit: blends the float32 `dest` row with that stream's message into the normed bf16 `out`.
fn control(c: *engine.Ctx, s: cuda.Stream, w: weights.Controller, dest: u64, stream: u32, norm: u64, out: u64) !void {
    const f = c.flow.r;
    try loop_kernels.controller(&c.loop, s, .{
        .dest = dest,
        .msg = f.msg + @as(u64, stream) * f32_row,
        .down = try present(w.down),
        .gain = try present(w.gain),
        .forget = try present(w.forget),
        .norm = norm,
        .h = f.gate_h,
        .out = out,
        .rows = rows,
    });
}

/// The restart's bank: the previous pass's two blocks, copied before this pass overwrites their slots (SPEC 9.1).
fn saveBank(c: *engine.Ctx, s: cuda.Stream, from: [2]u8) !void {
    const f = c.flow.r;
    if (from[1] < from[0] or from[1] + 1 - from[0] != bank.bank_slots or from[1] >= bank.stack_slots) return error.Inkling2BankSlots;
    for (f.bank, @as(usize, from[0])..) |dst, slot| {
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = dst, .len = bf16_row, .borrowed = true };
        try view.copyFrom(0, f.stack[slot], bf16_row, s.handle);
    }
}

/// Pass > 0 (SPEC 9.2, 9.3): the messages from the bank (to the trace, then the teacher's in their place: the bank is
/// the unforced stack, whose outlier channels the controllers' cancellations amplify), then the controllers blend the
/// carried float32 `dest` (the prelude destinations at a restart) with them into xq, xk, xv in `inp`. A local layer
/// broadcasts its one `dest` row and its one norm to the three streams; a global layer has a row and a norm for each.
fn attnInputs(c: *engine.Ctx, s: cuda.Stream, i: u32, v: visits.Visit, op: bank.Op, h: Hooks) !void {
    if (!op.messages) return;
    const f = c.flow.r;
    const loop = c.w.cfg.loop orelse return error.NotLoop;
    const lw = try layerWeights(c, v.layer);
    const qa = try present(c.w.cross_q orelse return error.Inkling2MissingWeight);
    const qkv = try present(c.w.cross_kv_q orelse return error.Inkling2MissingWeight);
    if (v.layer < loop.core_start or v.layer >= loop.core_end) return error.Inkling2LoopVisit;
    try loop_kernels.messages(&c.loop, s, f.bank, qa, qkv, v.layer - loop.core_start, f.msg, rows);
    if (h.trace) |tr| if (tr.msg) |put| {
        try s.synchronize();
        try put(tr.ctx, i, f.msg);
    };
    if (h.teacher) |te| if (te.msg) |get| if (get(te.ctx, i)) |vec| {
        if (vec.len != loop_kernels.n_streams * rows * hidden) return error.Inkling2TeacherShape;
        try upload(c, s, f.msg, std.mem.sliceAsBytes(vec));
        try s.synchronize(); // the host copy goes out of scope
    };

    const global = config.isGlobal(v.layer);
    if (op.restart and global) return error.Inkling2DestRows; // the prelude destinations have one row
    const dest = if (op.restart) f.dest else f.mix;
    const norms = try inputNorms(lw, v.layer);
    const ctl = try controllers(lw);
    for (0..attn_streams) |j| {
        const row: u32 = if (global) @intCast(j) else 0;
        try control(c, s, ctl[j], dest + @as(u64, row) * f32_row, @intCast(j), norms[row], f.inp + j * bf16_row);
    }

    const tr = h.trace orelse return;
    if (tr.xs) |put| {
        try loop_kernels.mixAdd(&c.loop, s, f.wide, f.inp, 0, attn_streams * rows * hidden);
        try s.synchronize();
        try put(tr.ctx, i, f.wide);
    }
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

/// Steps A-D of ref `_visit`: attention, the attention conv, the MLP input mix (blended with the mlp message at
/// pass > 0), the MoE, the MLP conv and the residual update.
fn sublayers(c: *engine.Ctx, s: cuda.Stream, stream: u32, pos: u64, i: u32, v: visits.Visit, op: bank.Op, h: Hooks) !void {
    const f = c.flow.r;
    const lw = try layerWeights(c, v.layer);
    const x: [3]u64 = if (op.messages or config.isGlobal(v.layer)) .{ f.inp, f.inp + bf16_row, f.inp + 2 * bf16_row } else .{ f.inp, f.inp, f.inp };
    try attn_visit.run(c, s, v, lw, x, rows, pos, stream, f.attn_pre);
    try conv(c, s, .{ .x = f.attn_pre, .state = c.cache.convAt(v.conv, stream, .attn), .w = try present(lw.attn_sconv), .out = f.attn_out, .c = hidden, .rows = rows, .mode = 0, .x_bf16 = 1 });

    const query = try present(c.w.attn_res.mlp[v.layer]);
    const norm = try present(lw.mlp_norm);
    const sources: []const u64 = if (op.first) f.stack[0 .. op.blocks + 1] else f.stack[0..op.blocks];
    const last = if (op.first) f.attn_out else f.resid;
    if (!op.first) try loop_kernels.mixAdd(&c.loop, s, f.resid, f.stack[op.blocks], f.attn_out, rows * hidden);
    if (op.messages) {
        try loop_kernels.mixF32(&c.loop, s, mixArgs(c, sources, last, op.first, query, no_norms, mlp_queries, f.mlp_dest));
        try control(c, s, (try controllers(lw))[mlp_stream], f.mlp_dest, mlp_stream, norm, f.mlp_in);
    } else {
        try ops.mix(&c.ops, s, mixArgs(c, sources, last, op.first, query, .{ norm, 0, 0 }, mlp_queries, f.mlp_in));
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

/// Step E: the vector the next visit reads: `inp` (bf16, normed) before a pass 0 visit and for the head, `mix` (the
/// fp32 `dest`) before a pass > 0 visit; before a restart the saved prelude destinations stand in `dest`.
fn nextInput(c: *engine.Ctx, s: cuda.Stream, i: u32, v: visits.Visit, op: bank.Op) !void {
    const f = c.flow.r;
    const blocks = op.blocks + @intFromBool(op.first);
    const sources = f.stack[0..blocks];
    switch (op.next) {
        .mix_norm => {
            const nv = c.sched.visits[i + 1];
            const nlw = try layerWeights(c, nv.layer);
            const nq = carriedRows(&c.sched, i);
            const query = try present(c.w.attn_res.attn[v.layer]);
            const norms = try inputNorms(nlw, nv.layer);
            try ops.mix(&c.ops, s, mixArgs(c, sources, f.stack[blocks], true, query, norms, nq, f.inp));
            if (op.save_dests) {
                if (nq != 1) return error.Inkling2DestRows;
                try loop_kernels.mixF32(&c.loop, s, mixArgs(c, sources, f.stack[blocks], true, query, no_norms, nq, f.dest));
            }
        },
        .mix_fp32 => {
            const query = try present(c.w.attn_res.attn[v.layer]);
            try loop_kernels.mixF32(&c.loop, s, mixArgs(c, sources, f.stack[blocks], true, query, no_norms, carriedRows(&c.sched, i), f.mix));
        },
        .head => {
            const query = try present(c.w.attn_res.output);
            try ops.mix(&c.ops, s, mixArgs(c, sources, f.stack[blocks], true, query, .{ try present(c.w.norm), 0, 0 }, head_queries, f.inp));
        },
        .prelude_dests => {},
    }
}

/// What the trace sees after a visit: the carried vector as float32 and its row count.
fn traced(c: *engine.Ctx, s: cuda.Stream, i: u32, op: bank.Op) !struct { ptr: u64, qn: u32 } {
    const f = c.flow.r;
    const qn = carriedRows(&c.sched, i);
    switch (op.next) {
        .prelude_dests => return .{ .ptr = f.dest, .qn = 1 },
        .mix_fp32 => return .{ .ptr = f.mix, .qn = qn },
        .mix_norm, .head => {
            try loop_kernels.mixAdd(&c.loop, s, f.wide, f.inp, 0, qn * rows * hidden);
            return .{ .ptr = f.wide, .qn = qn };
        },
    }
}

/// Replaces the carried input before visit `i + 1` with the teacher's vector (`qn` rows of float32).
fn teach(c: *engine.Ctx, s: cuda.Stream, op: bank.Op, qn: u32, vec: []const f32) !void {
    const f = c.flow.r;
    if (vec.len != qn * rows * hidden) return error.Inkling2TeacherShape;
    switch (op.next) {
        .prelude_dests => try upload(c, s, f.dest, std.mem.sliceAsBytes(vec)),
        .mix_fp32 => try upload(c, s, f.mix, std.mem.sliceAsBytes(vec)),
        .mix_norm, .head => {
            var bits: [global_inputs * hidden]u16 = undefined;
            for (vec, bits[0..vec.len]) |x, *b| b.* = bf16Bits(x);
            try upload(c, s, f.inp, std.mem.sliceAsBytes(bits[0..vec.len]));
        },
    }
    try s.synchronize(); // the host copy goes out of scope
}

/// Visits `0..last_visit` of the schedule for one token at position `pos` of `stream`. The carried vector after each
/// visit, each visit's MoE input and a pass > 0 visit's messages and controller outputs go to `h.trace`; `h.teacher`
/// replaces the carried vector, the messages and the MoE input, so errors cannot accumulate.
pub fn step(c: *engine.Ctx, s: cuda.Stream, stream: u32, token: u32, pos: u64, last_visit: u32, h: Hooks) !void {
    if (c.w.cfg.loop == null) return error.NotLoop;
    if (last_visit >= c.sched.n) return error.Inkling2VisitRange;
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
        if (op.bank_from) |from| try saveBank(c, s, from);
        try attnInputs(c, s, i, v, op, h);
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

test "layers read cover the next visit's input norms" {
    const m = try miniSchedule();
    try ts.expectEqual(@as(u32, 6), layersRead(&m.s, 5));
    try ts.expectEqual(@as(u32, 17), layersRead(&m.s, 16));
    try ts.expectEqual(@as(u32, 17), layersRead(&m.s, 17)); // the restart reads layer 6 again
    try ts.expectEqual(@as(u32, 17), layersRead(&m.s, 40));
    try ts.expectEqual(@as(u32, 18), layersRead(&m.s, 41)); // the coda's first input norm
    try ts.expectEqual(@as(u32, 23), layersRead(&m.s, 46));
    try ts.expectEqual(@as(u32, 23), layersRead(&m.s, 47));
}

test "the loop visits read the bank, the messages and the prelude destinations" {
    const m = try miniSchedule();
    const p = bank.plan(&m.s, &m.cfg);
    for (0..m.s.n) |i| {
        const v = m.s.visits[i];
        try ts.expectEqual(v.pass > 0, p[i].messages);
        // a pass > 0 visit reads the fp32 dest: from the previous visit's mix, or the prelude destinations at a restart
        if (i > 0 and v.pass > 0) try ts.expect(p[i - 1].next == (if (v.restart) bank.NextInput.prelude_dests else bank.NextInput.mix_fp32));
        if (v.restart) try ts.expect(!config.isGlobal(v.layer)); // the prelude destinations have one row
    }
    try ts.expect(p[47].next == .head);
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

test "step refuses visits past the schedule before touching the device" {
    var c: engine.Ctx = undefined;
    var w: weights.Weights = undefined;
    const m = try miniSchedule();
    w.cfg = m.cfg;
    c.w = &w;
    c.sched = m.s;
    var d: cuda.Driver = undefined;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try ts.expectError(error.Inkling2VisitRange, step(&c, s, 0, 0, 0, 48, .{}));
    try ts.expectError(error.Inkling2VisitRange, step(&c, s, 0, 0, 0, 63, .{}));
}
