//! Inkling 2 Small (model_type inkling2) on CUDA: config, chat, reply split, admission, kernels, EXL3 weights, forward.

pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const kernels = @import("kernels.zig");
pub const ops = @import("ops.zig");
pub const experts = @import("experts.zig");
pub const dspark_ops = @import("dspark_ops.zig");
pub const visits = @import("visits.zig");
pub const budget = @import("budget.zig");
pub const loop_kernels = @import("loop_kernels.zig");
pub const check_rows = @import("check_rows.zig");
pub const weights = @import("weights.zig");
pub const moe = @import("moe.zig");
pub const bank = @import("bank.zig");
pub const proc_status = @import("proc_status.zig");
pub const ablit = @import("ablit.zig");
pub const cache = @import("cache.zig");
pub const kv8 = @import("kv8.zig");
pub const plan = @import("plan.zig");
pub const config = @import("config.zig");
pub const engine = @import("engine.zig");
pub const attn_visit = @import("attn_visit.zig");
pub const decode = @import("decode.zig");
pub const head = @import("head.zig");
pub const prof = @import("prof.zig");
pub const prefill = @import("prefill.zig");
pub const prefill4 = @import("prefill4.zig");
pub const drafter = @import("drafter.zig");
pub const snap = @import("snap.zig");
pub const backend = @import("backend.zig");
pub const native = @import("native.zig");

/// config.json's name in the release.
pub const model_type = "inkling2";

test {
    _ = image;
    _ = vision;
    _ = kernels;
    _ = ops;
    _ = experts;
    _ = dspark_ops;
    _ = visits;
    _ = budget;
    _ = loop_kernels;
    _ = check_rows;
    _ = weights;
    _ = moe;
    _ = bank;
    _ = proc_status;
    _ = ablit;
    _ = cache;
    _ = kv8;
    _ = engine;
    _ = attn_visit;
    _ = prof;
    _ = decode;
    _ = head;
    _ = prefill;
    _ = drafter;
    _ = snap;
    _ = backend;
    _ = native;
}
