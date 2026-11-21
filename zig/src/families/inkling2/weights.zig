//! Inkling 2 Mini's text weights on the device, uploaded one layer at a time with the experts left packed.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const st = @import("safetensors");
const config = @import("config.zig");
const names = @import("names.zig");
const layer_plan = @import("plan.zig");
const visits = @import("visits.zig");
const budget = @import("budget.zig");

const Checkpoint = core.checkpoint.Checkpoint;

const llm_prefix = "model.llm.";
const layer_prefix = llm_prefix ++ "layers.";
const vision_prefix = "model.visual.";
const config_file = "config.json";
const config_max_bytes = 1 << 20;
const max_rank = 4;
const bad_log_limit = 5;

/// The streams of a layer's controllers, in the order of `names.controller_streams`.
const stream_order = [_][]const u8{ "attn", "k", "v", "mlp" };

/// A tensor on the device: its address, byte count and the checkpoint's dtype and shape.
pub const Dev = struct { ptr: u64, bytes: usize, dtype: st.DType, shape: [max_rank]u64, rank: u8 };

const empty: Dev = .{ .ptr = 0, .bytes = 0, .dtype = .u8, .shape = @splat(0), .rank = 0 };

/// One layer's routed experts, packed as stored. NVFP4 carries `*_amax` and `*_outer`; MXFP8 carries neither.
pub const Expert = struct {
    w13: Dev = empty,
    w13_scale: Dev = empty,
    w13_amax: ?Dev = null,
    w13_outer: ?Dev = null,
    w2: Dev = empty,
    w2_scale: Dev = empty,
    w2_amax: ?Dev = null,
    w2_outer: ?Dev = null,
};

pub const Mlp = struct { gate: Dev = empty, gate_bias: Dev = empty, experts: Expert = .{} };

/// A loop controller: the gate between passes of one stream.
pub const Controller = struct { down: Dev = empty, gain: Dev = empty, forget: Dev = empty };

/// A layer's tensors. A local layer has `attn_norm` and the `*_sconv` weights; a global one has the q, k and v input
/// norms, `static_w` in `k_sconv`/`v_sconv` and the data-dependent `*_sconv_proj`.
pub const LayerW = struct {
    wq_du: Dev = empty,
    wk_dv: Dev = empty,
    wv_dv: Dev = empty,
    wr_du: Dev = empty,
    wo_ud: Dev = empty,
    gate_attn: Dev = empty,
    attn_sink_proj: Dev = empty,
    q_l2_clip: Dev = empty,
    rel_l2_clip: Dev = empty,
    rel_logits_proj: Dev = empty,
    k_sconv: Dev = empty,
    v_sconv: Dev = empty,
    k_sconv_proj: ?Dev = null,
    v_sconv_proj: ?Dev = null,
    attn_norm: ?Dev = null,
    q_norm: ?Dev = null,
    k_norm: ?Dev = null,
    v_norm: ?Dev = null,
    attn_sconv: Dev = empty,
    mlp_norm: Dev = empty,
    mlp_sconv: Dev = empty,
    mlp: Mlp = .{},
    controllers: ?[stream_order.len]Controller = null, // stream order attn, k, v, mlp; core layers only
};

/// The attention-residual queries: `attn[j]` feeds the attention input of layer j + 1.
pub const AttnRes = struct {
    output: Dev = empty,
    mlp: [config.max_layers]Dev = @splat(empty),
    attn: [config.max_layers]Dev = @splat(empty),
};

const layer_keys = .{
    .{ "attn.wq_du.weight", "wq_du" },
    .{ "attn.wk_dv.weight", "wk_dv" },
    .{ "attn.wv_dv.weight", "wv_dv" },
    .{ "attn.wr_du.weight", "wr_du" },
    .{ "attn.wo_ud.weight", "wo_ud" },
    .{ "attn.gate_attn.weight", "gate_attn" },
    .{ "attn.attn_sink_proj.weight", "attn_sink_proj" },
    .{ "attn.q_l2_clip.scale", "q_l2_clip" },
    .{ "attn.rel_l2_clip.scale", "rel_l2_clip" },
    .{ "attn.rel_logits_proj.proj", "rel_logits_proj" },
    .{ "attn.k_sconv.weight", "k_sconv" },
    .{ "attn.k_sconv.static_w", "k_sconv" },
    .{ "attn.k_sconv.weight_proj", "k_sconv_proj" },
    .{ "attn.v_sconv.weight", "v_sconv" },
    .{ "attn.v_sconv.static_w", "v_sconv" },
    .{ "attn.v_sconv.weight_proj", "v_sconv_proj" },
    .{ "attn_norm.weight", "attn_norm" },
    .{ "q_input_norm.weight", "q_norm" },
    .{ "k_input_norm.weight", "k_norm" },
    .{ "v_input_norm.weight", "v_norm" },
    .{ "attn_sconv.weight", "attn_sconv" },
    .{ "mlp_norm.weight", "mlp_norm" },
    .{ "mlp_sconv.weight", "mlp_sconv" },
};

const gate_keys = .{
    .{ "mlp.gate.weight", "gate" },
    .{ "mlp.gate.bias", "gate_bias" },
};

const expert_keys = .{
    .{ "mlp.experts.w13_weight", "w13" },
    .{ "mlp.experts.w13_weight_scale", "w13_scale" },
    .{ "mlp.experts.w13_weight_amax", "w13_amax" },
    .{ "mlp.experts.w13_weight_outer_scale_denominator", "w13_outer" },
    .{ "mlp.experts.w2_weight", "w2" },
    .{ "mlp.experts.w2_weight_scale", "w2_scale" },
    .{ "mlp.experts.w2_weight_amax", "w2_amax" },
    .{ "mlp.experts.w2_weight_outer_scale_denominator", "w2_outer" },
};

const controller_prefix = "controllers.";
const controller_parts = .{
    .{ "down_weight", "down" },
    .{ "gain_weight", "gain" },
    .{ "forget_weight", "forget" },
};

const attn_res_output = "attn_res.output_query";
const attn_res_mlp = "attn_res.mlp_queries.";
const attn_res_attn = "attn_res.attn_queries.";

/// Device memory behind the loader: `alloc` returns an address, `upload` returns once the bytes landed, `free`
/// gives the address back, and `release` runs after each layer's tensors landed.
pub const Uploader = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        alloc: *const fn (ctx: *anyopaque, bytes: usize) anyerror!u64,
        upload: *const fn (ctx: *anyopaque, ptr: u64, bytes: []const u8) anyerror!void,
        free: *const fn (ctx: *anyopaque, ptr: u64, bytes: usize) void,
        release: *const fn (ctx: *anyopaque) void,
    };
};

fn deviceAlloc(ctx: *anyopaque, bytes: usize) anyerror!u64 {
    const d: *const cuda.Driver = @ptrCast(@alignCast(ctx));
    const b = try cuda.DeviceBuffer.alloc(d, bytes);
    return b.ptr;
}

fn deviceUpload(ctx: *anyopaque, ptr: u64, bytes: []const u8) anyerror!void {
    const d: *const cuda.Driver = @ptrCast(@alignCast(ctx));
    const b: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

fn deviceFree(ctx: *anyopaque, ptr: u64, bytes: usize) void {
    const d: *const cuda.Driver = @ptrCast(@alignCast(ctx));
    var b: cuda.DeviceBuffer = .{ .d = d, .ptr = ptr, .len = bytes };
    b.free();
}

// DeviceBuffer.upload blocks until the copy landed, so a layer boundary has nothing left to wait for.
fn deviceRelease(_: *anyopaque) void {}

const device_vtable: Uploader.VTable = .{ .alloc = deviceAlloc, .upload = deviceUpload, .free = deviceFree, .release = deviceRelease };

/// Uploads through `cuMemAlloc`; load before any sub-allocator hook is installed, since `free` assumes none owns the address.
pub fn deviceUploader(d: *const cuda.Driver) Uploader {
    std.debug.assert(cuda.memory.currentHook() == null);
    return .{ .ctx = @constCast(d), .vtable = &device_vtable };
}

const Held = struct { ptr: u64, bytes: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    up: Uploader,
    ck: Checkpoint,
    cfg: config.Config,
    prec: [config.max_layers]layer_plan.Precision = undefined,
    first: u32 = 0,
    last: u32 = 0, // layers first..last are uploaded
    layers: [config.max_layers]LayerW = @splat(.{}),
    embed: ?Dev = null,
    embed_host: ?[]const u8 = null, // bf16 [vocab, hidden] in the mapping; valid while the weights live
    unembed: Dev = empty,
    embed_norm: Dev = empty,
    norm: Dev = empty,
    attn_res: AttnRes = .{},
    cross_q: ?Dev = null,
    cross_kv_q: ?Dev = null,
    plan: budget.Budget = undefined,
    device_bytes: u64 = 0,
    held: std.ArrayList(Held) = .empty,
    index: std.StringArrayHashMapUnmanaged(Dev) = .empty,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(w: *Weights) void {
        for (w.held.items) |h| w.up.vtable.free(w.up.ctx, h.ptr, h.bytes);
        w.held.deinit(w.gpa);
        w.arena.deinit();
        w.ck.close();
        w.* = undefined;
    }

    /// An uploaded tensor by its name below `model.llm.` (`layers.0.attn.wo_ud.weight`).
    pub fn find(w: *const Weights, name: []const u8) ?Dev {
        return w.index.get(name);
    }

    /// The embedding rows of `ids` into `out` (`ids.len * hidden` bf16 words), read from the mapping.
    pub fn gatherEmbed(w: *const Weights, ids: []const u32, out: []u16) error{ NoHostEmbed, TokenOutOfRange, BadOutput }!void {
        const rows = w.embed_host orelse return error.NoHostEmbed;
        const hidden: usize = w.cfg.text.hidden;
        const row_bytes = hidden * @sizeOf(u16);
        if (out.len != ids.len * hidden) return error.BadOutput;
        for (ids, 0..) |id, i| {
            const at = @as(usize, id) * row_bytes;
            if (at + row_bytes > rows.len) return error.TokenOutOfRange;
            @memcpy(std.mem.sliceAsBytes(out[i * hidden ..][0..hidden]), rows[at..][0..row_bytes]);
        }
    }

    fn put(w: *Weights, item: Item, tensor: core.checkpoint.Tensor) !void {
        const ptr = try w.up.vtable.alloc(w.up.ctx, tensor.bytes.len);
        try w.held.append(w.gpa, .{ .ptr = ptr, .bytes = tensor.bytes.len });
        try w.up.vtable.upload(w.up.ctx, ptr, tensor.bytes);
        w.device_bytes += tensor.bytes.len;
        var dev: Dev = .{ .ptr = ptr, .bytes = tensor.bytes.len, .dtype = tensor.dtype, .shape = @splat(0), .rank = tensor.rank };
        for (tensor.shape[0..tensor.rank], 0..) |d, i| dev.shape[i] = d;
        try w.index.put(w.arena.allocator(), try w.arena.allocator().dupe(u8, item.llm_name), dev);
        try w.assign(item, dev);
    }

    fn assign(w: *Weights, item: Item, dev: Dev) !void {
        const l = item.layer orelse return w.assignGlobal(item.key, dev);
        return assignLayer(&w.layers[l], item.key, dev);
    }

    fn assignGlobal(w: *Weights, key: []const u8, dev: Dev) !void {
        if (std.mem.eql(u8, key, "embed.weight")) w.embed = dev else if (std.mem.eql(u8, key, "embed_norm.weight")) w.embed_norm = dev else if (std.mem.eql(u8, key, "norm.weight")) w.norm = dev else if (std.mem.eql(u8, key, "unembed.weight")) w.unembed = dev else if (std.mem.eql(u8, key, attn_res_output)) w.attn_res.output = dev else if (std.mem.eql(u8, key, "middle_cycle_cross_pass_queries")) w.cross_q = dev else if (std.mem.eql(u8, key, "middle_cycle_cross_pass_kv_queries")) w.cross_kv_q = dev else if (std.mem.startsWith(u8, key, attn_res_mlp)) {
            w.attn_res.mlp[try queryIndex(key, attn_res_mlp)] = dev;
        } else if (std.mem.startsWith(u8, key, attn_res_attn)) {
            w.attn_res.attn[try queryIndex(key, attn_res_attn)] = dev;
        } else return error.UnknownTensor;
    }
};

fn queryIndex(key: []const u8, prefix: []const u8) !usize {
    const i = std.fmt.parseInt(usize, key[prefix.len..], 10) catch return error.UnknownTensor;
    if (i >= config.max_layers) return error.UnknownTensor;
    return i;
}

fn assignLayer(l: *LayerW, key: []const u8, dev: Dev) !void {
    inline for (layer_keys) |kv| if (std.mem.eql(u8, key, kv[0])) {
        @field(l, kv[1]) = dev;
        return;
    };
    inline for (gate_keys) |kv| if (std.mem.eql(u8, key, kv[0])) {
        @field(l.mlp, kv[1]) = dev;
        return;
    };
    inline for (expert_keys) |kv| if (std.mem.eql(u8, key, kv[0])) {
        @field(l.mlp.experts, kv[1]) = dev;
        return;
    };
    if (std.mem.startsWith(u8, key, controller_prefix)) {
        const rest = key[controller_prefix.len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownTensor;
        const stream = for (stream_order, 0..) |s, i| {
            if (std.mem.eql(u8, s, rest[0..dot])) break i;
        } else return error.UnknownTensor;
        if (l.controllers == null) l.controllers = @splat(.{});
        inline for (controller_parts) |kv| if (std.mem.eql(u8, rest[dot + 1 ..], kv[0])) {
            @field(l.controllers.?[stream], kv[1]) = dev;
            return;
        };
    }
    return error.UnknownTensor;
}

/// One text tensor the map names: `key` is the name below `model.llm.` (and below `layers.N.` for a layer's).
const Item = struct {
    name: []const u8,
    llm_name: []const u8,
    key: []const u8,
    dtype: st.DType,
    dims: [max_rank]usize,
    rank: u8,
    layer: ?u32,
};

const Collect = struct {
    arena: std.mem.Allocator,
    items: std.ArrayList(Item) = .empty,

    pub fn put(c: *Collect, name: []const u8, dtype: st.DType, dims: []const u64) !void {
        if (std.mem.startsWith(u8, name, vision_prefix)) return;
        if (!std.mem.startsWith(u8, name, llm_prefix) or dims.len > max_rank) return error.WeightShape;
        const own = try c.arena.dupe(u8, name); // the map formats names into a buffer it reuses
        var item: Item = .{ .name = own, .llm_name = own[llm_prefix.len..], .key = own[llm_prefix.len..], .dtype = dtype, .dims = @splat(0), .rank = @intCast(dims.len), .layer = null };
        for (dims, 0..) |d, i| item.dims[i] = std.math.cast(usize, d) orelse return error.WeightShape;
        if (std.mem.startsWith(u8, name, layer_prefix)) {
            const rest = own[layer_prefix.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.WeightShape;
            item.layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return error.WeightShape;
            item.key = rest[dot + 1 ..];
        }
        try c.items.append(c.arena, item);
    }
};

fn readConfig(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !config.Config {
    const path = try std.fs.path.join(arena, &.{ dir, config_file });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(config_max_bytes));
    var why: config.Why = .{};
    return config.parse(arena, text, &why) catch |e| {
        std.log.err("{s}: {s}", .{ path, why.text() });
        return e;
    };
}

fn logList(label: []const u8, list: []const []const u8) void {
    for (list[0..@min(list.len, bad_log_limit)]) |n| std.log.warn("{s}: {s}", .{ label, n });
}

/// Loads the text weights of the checkpoint under `dir`: refuses (`WeightShape`) a tensor the weight map disagrees
/// with and (`OverBudget`) a budget above `limit` before any upload, then uploads one layer at a time.
pub fn load(gpa: std.mem.Allocator, io: std.Io, up: Uploader, dir: []const u8, o: budget.Options, limit: u64) !Weights {
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const a = scratch.allocator();

    var w: Weights = .{ .gpa = gpa, .up = up, .ck = try Checkpoint.openModel(gpa, io, dir), .cfg = undefined, .arena = std.heap.ArenaAllocator.init(gpa) };
    errdefer w.deinit();
    w.cfg = try readConfig(a, io, dir);
    const layers = w.cfg.text.layers;
    if (o.layers) |r| if (r.first > r.last or r.last >= layers) return error.BadLayerRange;
    w.first = if (o.layers) |r| r.first else 0;
    w.last = if (o.layers) |r| r.last else layers - 1;

    var header: st.Header = .empty;
    for (w.ck.files.items) |f| try names.merge(a, &header, f.names);
    try names.precisions(header, layers, &w.prec);
    const report = try names.check(a, &w.cfg, header, .model);
    if (!report.clean()) {
        logList("missing", report.missing.items);
        logList("wrong dtype or shape", report.wrong.items);
        logList("unknown", report.unknown.items);
        return error.WeightShape;
    }

    const precs = w.prec[0..layers];
    const lp = try layer_plan.build(&w.cfg.text, precs);
    const sched = try visits.build(&w.cfg, &lp);
    w.plan = budget.plan(&w.cfg, &sched, precs, o);
    try budget.check(w.plan, limit);

    var collect: Collect = .{ .arena = a };
    try names.all(&w.cfg, precs, .model, &collect);
    // globals first, then each layer on its own so its pages can go before the next layer's arrive
    try stage(&w, collect.items.items, null, o);
    for (w.first..w.last + 1) |l| {
        try stage(&w, collect.items.items, @intCast(l), o);
        w.ck.release();
        w.up.vtable.release(w.up.ctx);
    }
    return w;
}

fn stage(w: *Weights, items: []const Item, layer: ?u32, o: budget.Options) !void {
    for (items) |item| {
        if (item.layer != layer) continue;
        const tensor = try w.ck.expect(item.name, item.dtype, item.dims[0..item.rank]);
        if (o.embed_host and item.layer == null and std.mem.eql(u8, item.key, "embed.weight")) {
            w.embed_host = tensor.bytes;
            continue;
        }
        try w.put(item, tensor);
        // the unembedding is about 0.9 GiB: its pages go before the next global arrives
        if (layer == null) w.ck.release();
    }
}

const t = std.testing;
const config_test = @import("config_test.zig");

/// Tiny-random's config with a loop (core layers 6..12 of 12, two passes) and a vocabulary small enough for a test file.
fn tinyLoopConfig(a: std.mem.Allocator) ![]u8 {
    const edits = [_][2][]const u8{
        .{ "\"model_type\": \"inkling2\"", "\"model_type\": \"inkling2_loop\"" },
        .{ "Inkling2ForConditionalGeneration", "Inkling2LoopForConditionalGeneration" },
        .{ "\"use_gate_bias\": true,\n    \"tie_word_embeddings\": false", "\"use_gate_bias\": true,\n    \"tie_word_embeddings\": false,\n    \"loop_config\": {\"core_start\": 6, \"core_end\": 12, \"num_passes\": 2, \"gate_rank\": 8, \"is_varloop\": false}" },
        .{ "201024", "4096" },
        .{ "200058", "4000" },
        .{ "200006", "4006" },
        .{ "200054", "4054" },
    };
    var text: []u8 = try a.dupe(u8, config_test.tiny);
    for (edits) |e| text = try std.mem.replaceOwned(u8, a, text, e[0], e[1]);
    return text;
}

fn dtypeTag(d: st.DType) []const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f32 => "F32",
        .f8_e4m3 => "F8_E4M3",
        .u8 => "U8",
        else => unreachable,
    };
}

/// A tensor's bytes in the synthetic file: random from its name, so a test can recompute them.
fn fill(name: []const u8, out: []u8) void {
    var rng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(0, name));
    rng.random().bytes(out);
}

const Spec = struct { name: []const u8, dtype: st.DType, dims: []const u64 };

const Gen = struct {
    arena: std.mem.Allocator,
    specs: std.ArrayList(Spec) = .empty,

    pub fn put(g: *Gen, name: []const u8, dtype: st.DType, dims: []const u64) !void {
        try g.specs.append(g.arena, .{ .name = try g.arena.dupe(u8, name), .dtype = dtype, .dims = try g.arena.dupe(u64, dims) });
    }
};

const first_mxfp8_layer = 8;

/// A model folder in a temp dir: tiny-random's tensors with layers from `first_mxfp8_layer` in MXFP8. `swap` names one
/// tensor whose first two dims are written swapped (the bytes still fit).
const Fixture = struct {
    tmp: std.testing.TmpDir,
    dir: []u8,
    cfg: config.Config,

    fn init(a: std.mem.Allocator, swap: ?[]const u8) !Fixture {
        var f: Fixture = .{ .tmp = std.testing.tmpDir(.{}), .dir = undefined, .cfg = undefined };
        errdefer f.tmp.cleanup();
        f.dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{&f.tmp.sub_path});
        const json = try tinyLoopConfig(a);
        try f.tmp.dir.writeFile(t.io, .{ .sub_path = config_file, .data = json });
        var why: config.Why = .{};
        f.cfg = try config.parse(a, json, &why);

        var prec: [config.max_layers]layer_plan.Precision = undefined;
        for (0..f.cfg.text.layers) |i| prec[i] = if (i < first_mxfp8_layer) .nvfp4 else .mxfp8;
        var gen: Gen = .{ .arena = a };
        try names.all(&f.cfg, prec[0..f.cfg.text.layers], .model, &gen);

        var header: std.ArrayList(u8) = .empty;
        var data: std.ArrayList(u8) = .empty;
        try header.append(a, '{');
        for (gen.specs.items, 0..) |s, i| {
            var n: usize = s.dtype.size();
            for (s.dims) |d| n *= d;
            const begin = data.items.len;
            try data.resize(a, begin + n);
            fill(s.name, data.items[begin..]);
            try header.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", s.name, dtypeTag(s.dtype) });
            const flip = swap != null and std.mem.eql(u8, swap.?, s.name);
            for (s.dims, 0..) |_, k| {
                const at = if (flip and k < 2) 1 - k else k;
                try header.print(a, "{s}{d}", .{ if (k == 0) "" else ",", s.dims[at] });
            }
            try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ begin, begin + n });
        }
        try header.append(a, '}');
        var file: std.ArrayList(u8) = .empty;
        try file.appendNTimes(a, 0, 8);
        std.mem.writeInt(u64, file.items[0..8], header.items.len, .little);
        try file.appendSlice(a, header.items);
        try file.appendSlice(a, data.items);
        try f.tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = file.items });
        return f;
    }

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
    }
};

/// A host-side stand-in for the device: buffers in memory, counts of layer releases and live bytes.
const Fake = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(struct { ptr: u64, data: []u8 }) = .empty,
    next: u64 = 0x1000,
    releases: usize = 0,
    live: usize = 0,
    allocs: usize = 0,

    const vtable: Uploader.VTable = .{ .alloc = alloc, .upload = upload, .free = free, .release = release };

    fn uploader(f: *Fake) Uploader {
        return .{ .ctx = f, .vtable = &vtable };
    }

    fn self(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }

    fn alloc(ctx: *anyopaque, bytes: usize) anyerror!u64 {
        const f = self(ctx);
        const data = try f.gpa.alloc(u8, bytes);
        const ptr = f.next;
        f.next += (bytes + 255) / 256 * 256 + 256;
        try f.bufs.append(f.gpa, .{ .ptr = ptr, .data = data });
        f.live += bytes;
        f.allocs += 1;
        return ptr;
    }

    fn upload(ctx: *anyopaque, ptr: u64, bytes: []const u8) anyerror!void {
        const f = self(ctx);
        for (f.bufs.items) |b| if (b.ptr == ptr) {
            if (b.data.len != bytes.len) return error.ShortBuffer;
            @memcpy(b.data, bytes);
            return;
        };
        return error.UnknownAddress;
    }

    fn free(ctx: *anyopaque, ptr: u64, bytes: usize) void {
        const f = self(ctx);
        for (f.bufs.items) |*b| if (b.ptr == ptr) {
            std.debug.assert(b.data.len == bytes);
            f.gpa.free(b.data);
            b.data = &.{};
            f.live -= bytes;
            return;
        };
        unreachable;
    }

    fn release(ctx: *anyopaque) void {
        self(ctx).releases += 1;
    }

    fn deinit(f: *Fake) void {
        for (f.bufs.items) |b| f.gpa.free(b.data);
        f.bufs.deinit(f.gpa);
    }

    fn view(f: *const Fake, d: Dev) []const u8 {
        for (f.bufs.items) |b| if (b.ptr == d.ptr) return b.data;
        return &.{};
    }
};

const test_limit: u64 = 1 << 40;
const test_options: budget.Options = .{ .context = 512, .streams = 1, .chunk_rows = 16, .embed_host = false };

/// `d` holds the file's bytes for `name`, with that dtype and these dims.
fn expectTensor(f: *const Fake, d: Dev, name: []const u8, dtype: st.DType, dims: []const u64) !void {
    try t.expectEqual(dtype, d.dtype);
    try t.expectEqual(dims.len, d.rank);
    try t.expectEqualSlices(u64, dims, d.shape[0..d.rank]);
    const want = try t.allocator.alloc(u8, d.bytes);
    defer t.allocator.free(want);
    fill(name, want);
    try t.expectEqualSlices(u8, want, f.view(d));
}

test "loads a synthetic checkpoint" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var fx = try Fixture.init(arena.allocator(), null);
    defer fx.deinit();
    var fake: Fake = .{ .gpa = t.allocator };
    defer fake.deinit();

    var w = try load(t.allocator, t.io, fake.uploader(), fx.dir, test_options, test_limit);
    defer w.deinit();
    const h: u64 = fx.cfg.text.hidden;
    const m: u64 = fx.cfg.text.expert_inter;
    const e: u64 = fx.cfg.text.experts;

    try t.expectEqual(w.plan.weights, w.device_bytes);
    try t.expectEqual(@as(usize, w.device_bytes), fake.live);

    const l0 = w.layers[0];
    try expectTensor(&fake, l0.wo_ud, "model.llm.layers.0.attn.wo_ud.weight", .bf16, &.{ h, @as(u64, fx.cfg.text.heads(0).q) * fx.cfg.text.heads(0).dim });
    try expectTensor(&fake, w.layers[3].mlp.experts.w13, "model.llm.layers.3.mlp.experts.w13_weight", .u8, &.{ e, 2 * m, h / 2 });
    try expectTensor(&fake, w.layers[3].mlp.experts.w13_scale, "model.llm.layers.3.mlp.experts.w13_weight_scale", .f8_e4m3, &.{ e, 2 * m, h / 16 });
    try expectTensor(&fake, w.layers[3].mlp.experts.w13_amax.?, "model.llm.layers.3.mlp.experts.w13_weight_amax", .f32, &.{e});
    try expectTensor(&fake, w.layers[3].mlp.experts.w2_outer.?, "model.llm.layers.3.mlp.experts.w2_weight_outer_scale_denominator", .f32, &.{1});
    try expectTensor(&fake, w.layers[9].mlp.experts.w13, "model.llm.layers.9.mlp.experts.w13_weight", .f8_e4m3, &.{ e, 2 * m, h });
    try expectTensor(&fake, w.layers[9].mlp.experts.w2_scale, "model.llm.layers.9.mlp.experts.w2_weight_scale", .u8, &.{ e, h, m / 32 });
    try expectTensor(&fake, w.attn_res.attn[4], "model.llm.attn_res.attn_queries.4", .f32, &.{ 3, h });
    try expectTensor(&fake, w.unembed, "model.llm.unembed.weight", .bf16, &.{ fx.cfg.text.vocab, h });
    try expectTensor(&fake, w.layers[5].k_sconv_proj.?, "model.llm.layers.5.attn.k_sconv.weight_proj", .bf16, &.{ 4 * config.conv_taps, h });
    try t.expect(w.layers[0].k_sconv_proj == null and w.layers[0].attn_norm != null);
    try t.expect(w.layers[5].attn_norm == null and w.layers[5].q_norm != null);

    for (0..fx.cfg.text.layers) |i| {
        const ex = w.layers[i].mlp.experts;
        const nvfp4 = i < first_mxfp8_layer;
        try t.expectEqual(nvfp4, ex.w13_amax != null and ex.w2_amax != null and ex.w13_outer != null and ex.w2_outer != null);
        const core_layer = i >= fx.cfg.loop.?.core_start and i < fx.cfg.loop.?.core_end;
        try t.expectEqual(core_layer, w.layers[i].controllers != null);
    }
    const cc = w.layers[7].controllers.?;
    try expectTensor(&fake, cc[3].forget, "model.llm.layers.7.controllers.mlp.forget_weight", .bf16, &.{ h, 8 });
    try expectTensor(&fake, w.cross_kv_q.?, "model.llm.middle_cycle_cross_pass_kv_queries", .f32, &.{ 2 * 6, h });
    try t.expect(w.embed != null and w.embed_host == null);
    try t.expectEqual(w.layers[0].wo_ud.ptr, w.find("layers.0.attn.wo_ud.weight").?.ptr);
    try t.expect(w.find("layers.0.nothing") == null);
}

test "refuses a wrong shape before any upload" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var fx = try Fixture.init(arena.allocator(), "model.llm.layers.2.attn.wq_du.weight");
    defer fx.deinit();
    var fake: Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    try t.expectError(error.WeightShape, load(t.allocator, t.io, fake.uploader(), fx.dir, test_options, test_limit));
    try t.expectEqual(@as(usize, 0), fake.allocs);
}

test "release called per layer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var fx = try Fixture.init(arena.allocator(), null);
    defer fx.deinit();
    var fake: Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var w = try load(t.allocator, t.io, fake.uploader(), fx.dir, test_options, test_limit);
    try t.expectEqual(@as(usize, fx.cfg.text.layers), fake.releases);
    w.deinit();
    try t.expectEqual(@as(usize, 0), fake.live);

    var staged = test_options;
    staged.layers = .{ .first = 2, .last = 4 };
    fake.releases = 0;
    var s = try load(t.allocator, t.io, fake.uploader(), fx.dir, staged, test_limit);
    defer s.deinit();
    try t.expectEqual(@as(usize, 3), fake.releases);
    try t.expectEqual(s.plan.weights, s.device_bytes);
    try t.expect(s.find("layers.1.attn.wo_ud.weight") == null);
    try t.expect(s.find("layers.3.attn.wo_ud.weight") != null);
    try t.expect(s.find("embed.weight") != null);
}

test "a budget above the limit is refused before any upload" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var fx = try Fixture.init(arena.allocator(), null);
    defer fx.deinit();
    var fake: Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    try t.expectError(error.OverBudget, load(t.allocator, t.io, fake.uploader(), fx.dir, test_options, 1 << 20));
    try t.expectEqual(@as(usize, 0), fake.allocs);
    var bad = test_options;
    bad.layers = .{ .first = 3, .last = 12 };
    try t.expectError(error.BadLayerRange, load(t.allocator, t.io, fake.uploader(), fx.dir, bad, test_limit));
}

test "host embedding stays in the mapping and gathers rows" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var fx = try Fixture.init(arena.allocator(), null);
    defer fx.deinit();
    var fake: Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var host = test_options;
    host.embed_host = true;
    var w = try load(t.allocator, t.io, fake.uploader(), fx.dir, host, test_limit);
    defer w.deinit();
    try t.expect(w.embed == null and w.find("embed.weight") == null);
    try t.expectEqual(w.plan.weights, w.device_bytes);

    const hidden: usize = fx.cfg.text.hidden;
    var table: [4096 * 2 * 128]u8 = undefined;
    fill("model.llm.embed.weight", &table);
    const ids = [_]u32{ 7, 0, 3999 };
    var out: [3 * 128]u16 = undefined;
    try w.gatherEmbed(&ids, &out);
    for (ids, 0..) |id, i| {
        try t.expectEqualSlices(u8, table[id * hidden * 2 ..][0 .. hidden * 2], std.mem.sliceAsBytes(out[i * hidden ..][0..hidden]));
    }
    try t.expectError(error.TokenOutOfRange, w.gatherEmbed(&.{4096}, out[0..hidden]));
    try t.expectError(error.BadOutput, w.gatherEmbed(&ids, out[0..hidden]));
}
