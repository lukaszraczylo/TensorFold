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
pub const proc_status = @import("proc_status.zig");

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
    _ = proc_status;
}
