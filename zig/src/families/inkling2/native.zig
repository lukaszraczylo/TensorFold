//! Inkling 2 Mini (looped, ModelOpt NVFP4 and MXFP8) behind the native CUDA registry: plain decoding, no prompt cache.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const platform = @import("sys");
const budget = @import("budget.zig");
const cache = @import("cache.zig");
const config = @import("config.zig");
const engine = @import("engine.zig");
const plan = @import("plan.zig");
const visits = @import("visits.zig");
const weights = @import("weights.zig");
const be = @import("backend.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 65536;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;
/// A request's last verify window: one row.
pub const window_rows: u32 = be.window_rows;
pub const explains_open = true;
/// No snap_* hooks: the host plans no prompt cache and refuses --prompt-cache-gib by name.
pub const keeps_no_prompts = true;

const embed_host_env = "TF_INK2_EMBED_HOST";
const ablit_env = "TF_INK2_ABLIT";

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// No `precision` field: the server refuses --precision for this family by name.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *be.Inkling,
};

const Owned = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    w: weights.Weights,
    sched: visits.Schedule,
    cache: cache.Cache,
    c: engine.Ctx,
    ib: be.Inkling,
};

/// Opens the model with `o.lanes` streams, or (`lanes_fixed` off) the most of them that fit the room; the plan refuses first.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const embed_host = try platform.environment.read(bool, embed_host_env, false);
    const room: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.stream = try cuda.Stream.init(ctx.d, true);
    errdefer own.stream.deinit();
    var n: u32 = @max(o.lanes, 1);
    while (true) {
        const wo: budget.Options = .{ .context = o.context, .streams = n, .chunk_rows = prompt_rows, .embed_host = embed_host };
        own.w = weights.load(gpa, io, weights.deviceUploader(ctx.d), dir, wo, room) catch |e| {
            if (e != error.OverBudget or o.lanes_fixed or n == 1) return e;
            n -= 1;
            continue;
        };
        break;
    }
    errdefer own.w.deinit();
    const text = &own.w.cfg.text;
    const layers = try plan.build(text, own.w.prec[0..text.layers]);
    own.sched = try visits.build(&own.w.cfg, &layers);
    const co: cache.Options = .{ .context = o.context, .streams = n, .chunk_rows = prompt_rows };
    own.cache = try cache.Cache.init(ctx.d, &own.sched, text, co);
    errdefer own.cache.deinit();
    own.c.init(gpa, io, ctx.d, &own.w, own.sched, &own.cache, co) catch |e| return if (e == error.FileNotFound) error.AblitFileMissing else e; // the only file `Ctx.init` reads
    errdefer own.c.deinit();
    own.ib = try be.Inkling.init(gpa, &own.c, own.stream, n, o.context);
    errdefer own.ib.deinit();
    return .{
        .backend = own.ib.backend(),
        .facts = own.ib.facts(),
        .rows = window_rows,
        .stream_bytes = @intCast(cache.bytes(&own.sched, text, .{ .context = o.context, .streams = 1, .chunk_rows = prompt_rows })),
        .free_streams = n,
        .ctx = own,
        .deinit = release,
        .target = &own.ib,
    };
}

/// Whether a request could start now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Owned = @ptrCast(@alignCast(p));
    return own.ib.admits(prompt, max_tokens);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Inkling 2 Mini on CUDA has no draft head: it decodes one token a step",
        error.InklingTextOnly => "Inkling 2 Mini on CUDA reads text only: this request has images",
        error.StructuresUnsupported => "structured output is not built for Inkling 2 Mini on CUDA yet",
        error.WindowTooWide => "Inkling 2 Mini on CUDA verifies one row a window: it takes no drafts",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.CapturedKernels => "Inkling 2 Mini runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.OverBudget => "the CUDA memory budget does not hold Inkling 2 Mini's weights, caches and scratch at this --context and --parallel: lower either one, set TF_INK2_EMBED_HOST=1 to read the embedding rows from the host, or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_CUDA_MEMORY_LIMIT_GB",
        error.InvalidEnvironment => "a TF_INK2_* setting is invalid: TF_INK2_EMBED_HOST is 0 or 1, TF_INK2_ABLIT a file path (or 0), TF_INK2_ABLIT_SCALE a number above 0 and up to 16",
        error.AblitFileMissing => "TF_INK2_ABLIT names a file that does not exist: give the path of the TFINK2A1 edit file, or unset it",
        error.BadSize => "TF_INK2_ABLIT: the edit file's size does not match its header or this model",
        error.BadMagic => "TF_INK2_ABLIT: the file is not a TFINK2A1 edit file",
        error.BadShape => "TF_INK2_ABLIT: the edit file's hidden size, layer count or rank does not fit this model",
        error.NotFinite => "TF_INK2_ABLIT: the edit file holds a value that is not finite",
        error.NegativeScale => "TF_INK2_ABLIT: the edit file holds a negative scale",
        error.NotOrthonormal => "TF_INK2_ABLIT: the edit file's plane is not orthonormal",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.ib.deinit();
    own.c.deinit();
    own.cache.deinit();
    own.w.deinit();
    own.stream.deinit();
    own.gpa.destroy(own);
}

test "every refusal of its own is worded, the rest left to the server" {
    for ([_]anyerror{ error.PastContext, error.NoFreeSlot, error.NoDraftHead, error.InklingTextOnly, error.StructuresUnsupported, error.NonFiniteLogits, error.CapturedKernels, error.OverBudget, error.BadSize, error.BadMagic, error.BadShape, error.NotFinite, error.NegativeScale, error.NotOrthonormal, error.AblitFileMissing, error.WindowTooWide, error.InvalidEnvironment }) |e| {
        try std.testing.expect(explain(null, e) != null);
    }
    const over = explain(null, error.OverBudget).?;
    for ([_][]const u8{ "--context", "--parallel", embed_host_env }) |name| try std.testing.expect(std.mem.indexOf(u8, over, name) != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.AblitFileMissing).?, ablit_env) != null);
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}

test "fits" {
    var own: Owned = undefined;
    var free = [_]bool{ true, true };
    own.ib.free = &free;
    own.ib.context = 4096;
    try std.testing.expect(fits(&own, 100, 100));
    try std.testing.expect(!fits(&own, 4090, 100));
    free = .{ false, false };
    try std.testing.expect(!fits(&own, 100, 100));
}

test "the registry entry is the loop family's, over the ModelOpt NVFP4 format" {
    try std.testing.expectEqualStrings(config.model_type_loop, model_type);
    try std.testing.expectEqual(@as(usize, 1), formats.len);
    try std.testing.expectEqualStrings("modelopt-nvfp4", formats[0]);
    try std.testing.expectEqual(@as(u32, 512), prompt_rows);
}
