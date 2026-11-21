//! Residual stack, message bank and prelude restart plan: which stack slots each visit reads, and what it feeds the next visit.
const std = @import("std");
const config = @import("config.zig");
const visits = @import("visits.zig");

/// bf16 [rows, hidden] buffers: the embedding, three finished blocks and one partial.
pub const stack_slots = 5;
/// The two blocks a restart saves for the next pass's messages.
pub const bank_slots = 2;

pub const NextInput = enum { mix_norm, mix_fp32, prelude_dests, head };

pub const Op = struct {
    /// Finished blocks at the visit start (after a restart reset); the partial sits in slot `blocks`.
    blocks: u32,
    /// The layer closes the partial as a block and starts a new partial in slot `blocks + 1`.
    first: bool,
    /// Layer `core_start` of a later pass: copy `bank_from` to the bank, then continue from the prelude state.
    restart: bool,
    /// The visit before the core computes the prelude destinations once.
    save_dests: bool,
    /// First and last stack slot (inclusive) copied into the bank buffers.
    bank_from: ?[2]u8,
    /// The visit reads the cross-pass messages (pass > 0).
    messages: bool,
    next: NextInput,
};

/// One pass over the schedule keeping the finished-block count, as `ref/model.py` forward, `_visit` and `_next_input` do.
pub fn plan(s: *const visits.Schedule, c: *const config.Config) [visits.max_visits]Op {
    const loop = c.loop.?;
    var ops: [visits.max_visits]Op = undefined;
    var blocks: u32 = 0;
    var prelude_blocks: u32 = 0;
    for (s.visits[0..s.n], 0..) |v, i| {
        var bank_from: ?[2]u8 = null;
        if (v.layer == loop.core_start and v.pass == 0) prelude_blocks = blocks;
        if (v.restart) {
            std.debug.assert(blocks + 1 - (prelude_blocks + 1) == bank_slots);
            bank_from = .{ @intCast(prelude_blocks + 1), @intCast(blocks) };
            blocks = prelude_blocks;
        }
        const first = v.layer % c.text.block == 0;
        const next: NextInput = if (i + 1 == s.n)
            .head
        else if (s.visits[i + 1].restart)
            .prelude_dests
        else if (s.visits[i + 1].pass > 0)
            .mix_fp32
        else
            .mix_norm;
        ops[i] = .{
            .blocks = blocks,
            .first = first,
            .restart = v.restart,
            .save_dests = v.layer + 1 == loop.core_start and v.pass == 0,
            .bank_from = bank_from,
            .messages = v.pass > 0,
            .next = next,
        };
        if (first) blocks += 1;
        std.debug.assert(blocks < stack_slots);
    }
    return ops;
}

const t = std.testing;
const plan_mod = @import("plan.zig");
const config_test = @import("config_test.zig");

const Mini = struct {
    cfg: config.Config,
    p: plan_mod.Plan,
    s: visits.Schedule,

    fn init() !Mini {
        var why: config.Why = .{};
        const cfg = try config.parse(t.allocator, config_test.mini, &why);
        const prec: [config.max_layers]plan_mod.Precision = @splat(.nvfp4);
        const p = try plan_mod.build(&cfg.text, prec[0..cfg.text.layers]);
        return .{ .cfg = cfg, .p = p, .s = try visits.build(&cfg, &p) };
    }
};

test "bank restart indices" {
    const m = try Mini.init();
    const ops = plan(&m.s, &m.cfg);
    try t.expectEqual(@as(u32, 1), ops[6].blocks);
    try t.expect(ops[6].first and !ops[6].restart and !ops[6].messages and !ops[6].save_dests);
    try t.expect(ops[5].save_dests);
    for ([_]usize{ 18, 30 }) |i| {
        try t.expectEqual(@as(u32, 1), ops[i].blocks);
        try t.expect(ops[i].first and ops[i].restart and ops[i].messages);
        try t.expectEqual(@as(?[2]u8, .{ 2, 3 }), ops[i].bank_from);
    }
    try t.expectEqual(@as(?[2]u8, null), ops[6].bank_from);
    try t.expectEqual(@as(u32, 2), ops[12].blocks);
    try t.expect(ops[12].first);
    try t.expectEqual(@as(u32, 2), ops[24].blocks);
    try t.expect(ops[24].first);
    try t.expectEqual(@as(u32, 3), ops[17].blocks);
    try t.expect(!ops[17].first);
    try t.expectEqual(@as(u32, 3), ops[42].blocks);
    try t.expect(ops[42].first);
    try t.expectEqual(@as(u32, 4), ops[47].blocks);
    try t.expect(!ops[47].first);
}

test "next input kinds" {
    const m = try Mini.init();
    const ops = plan(&m.s, &m.cfg);
    try t.expectEqual(NextInput.mix_norm, ops[5].next);
    try t.expectEqual(NextInput.prelude_dests, ops[17].next);
    try t.expectEqual(NextInput.prelude_dests, ops[29].next);
    try t.expectEqual(NextInput.mix_norm, ops[41].next);
    try t.expectEqual(NextInput.mix_norm, ops[11].next);
    try t.expectEqual(NextInput.mix_fp32, ops[19].next);
    try t.expectEqual(NextInput.head, ops[47].next);
    for (0..47) |i| try t.expectEqual(i == 17 or i == 29, ops[i].next == .prelude_dests);
}

test "first visits" {
    const m = try Mini.init();
    const ops = plan(&m.s, &m.cfg);
    const want = [_]usize{ 0, 6, 12, 18, 24, 30, 36, 42 };
    for (0..m.s.n) |i| try t.expectEqual(std.mem.indexOfScalar(usize, &want, i) != null, ops[i].first);
}

test "messages" {
    const m = try Mini.init();
    const ops = plan(&m.s, &m.cfg);
    for (0..m.s.n) |i| try t.expectEqual(i >= 18 and i < 42, ops[i].messages);
}

// Expected table, derived from ref/model.py forward() and _visit(): a list model keeps blocks and partial as lists of
// tensor ids. The plan then runs on five physical slots; at every visit the slots must hold exactly the list model's
// blocks, partial and (after a restart) bank. This checks the slot arithmetic against the reference's list semantics.
test "plan reproduces the reference lists on physical slots for all 48 visits" {
    const m = try Mini.init();
    const ops = plan(&m.s, &m.cfg);
    const loop = m.cfg.loop.?;

    var list_blocks: [stack_slots]u32 = undefined;
    var nb: usize = 0;
    var partial: u32 = 0; // id 0 is the embedding
    var next_id: u32 = 1;
    var pre_blocks: [stack_slots]u32 = undefined;
    var pre_n: usize = 0;
    var pre_partial: u32 = 0;

    var slots: [stack_slots]u32 = @splat(std.math.maxInt(u32));
    var bank: [bank_slots]u32 = undefined;
    slots[0] = 0; // embedding in slot 0, the partial

    for (m.s.visits[0..m.s.n], 0..) |v, i| {
        var want_bank: ?[bank_slots]u32 = null;
        if (v.layer == loop.core_start) {
            if (v.pass == 0) {
                pre_n = nb;
                @memcpy(pre_blocks[0..nb], list_blocks[0..nb]);
                pre_partial = partial;
            } else {
                var full: [stack_slots]u32 = undefined;
                @memcpy(full[0..nb], list_blocks[0..nb]);
                full[nb] = partial;
                const skip = pre_n + 1;
                try t.expectEqual(@as(usize, bank_slots), nb + 1 - skip);
                want_bank = .{ full[skip], full[skip + 1] };
                nb = pre_n;
                @memcpy(list_blocks[0..nb], pre_blocks[0..nb]);
                partial = pre_partial;
            }
        }
        const op = ops[i];
        if (op.bank_from) |bf| {
            bank = .{ slots[bf[0]], slots[bf[1]] };
            try t.expectEqualSlices(u32, &want_bank.?, &bank);
        } else try t.expect(want_bank == null);
        // plan's block count: the reference's count at the visit start, after a restart's reset
        try t.expectEqual(@as(u32, @intCast(nb)), op.blocks);
        if (op.restart) try t.expectEqual(pre_n, op.blocks);
        // physical stack equals the list model's stack
        try t.expectEqualSlices(u32, list_blocks[0..nb], slots[0..nb]);
        try t.expectEqual(partial, slots[nb]);

        const first = v.layer % m.cfg.text.block == 0;
        const out = next_id;
        next_id += 1;
        if (first) {
            list_blocks[nb] = partial;
            nb += 1;
            partial = out;
            slots[op.blocks + 1] = out;
        } else {
            partial = out;
            slots[op.blocks] = out;
        }
    }
    try t.expectEqual(@as(usize, 4), nb);
}
