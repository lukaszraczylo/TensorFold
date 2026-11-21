//! `tf-ink2-check <subcommand> --out DIR [options]`: rows of Inkling 2 Mini's CUDA engine for a reference to compare.
const std = @import("std");
const cuda = @import("cuda");
const ink2 = @import("inkling2");
const environment = @import("sys").environment;

const usage =
    \\usage: tf-ink2-check <subcommand> --out DIR [options]
    \\smoke: fills a device buffer with tf_ink2l_fill and writes DIR/smoke.u32 (1024 words of 0xC0FFEE)
    \\actq --in DIR: reads x_nvfp4_<k>.bf16 and x_mxfp8_<k>.bf16 (shapes in DIR/shapes.json), writes nvfp4_<k>.f32 and mxfp8_<k>.f32 (dequantised), nvfp4_values_/nvfp4_scales_/nvfp4_amax_<k>.f32, mxfp8_values_<k>.f32 and mxfp8_exps_<k>.u8
    \\moe --model DIR --layers L-L --rows N --in DIR: reads DIR/x.bf16 ([N, 2304]), runs layer L's routed MoE, writes out.bf16, pick.i32 ([N, 8]) and gamma.f32 ([N, 8])
    \\loop --model DIR --layer L --rows N --in DIR: reads bank.bf16 ([2, N, 2304]), dest.f32 ([N, 2304]) and norm.bf16 ([2304]), runs core layer L's four cross-pass messages and controllers, writes msgs.f32 ([4, N, 2304]) and ctl_<attn|k|v|mlp>.bf16 ([N, 2304])
    \\ablit --file F --layer L --mult M --in DIR: reads DIR/y.bf16 ([N, 2304]), applies the edit file F at layer L with scale multiplier M, writes y.bf16
    \\attn --model DIR --layer L --pass P --n N --rows R [--pos0 P0] --in DIR: reads DIR/xs.bf16 ([N, 3, 2304]: xq, xk, xv a step), runs the visit (L, P)'s attention on them R rows a call from position P0 with the visit's caches carried (P0 > 0: k0.bf16 and v0.bf16 [keep, 8, 128] fill the rows before it), writes out.bf16 ([N, 2304], the wo_ud projection)
    \\weights --model DIR [--layers A-B] [--readback NAME,...]: uploads the text weights (TF_INK2_EMBED_HOST=1 keeps the
    \\    embedding on the host), writes DIR/weights.json with peak_rss_mib and device_bytes, and one row per readback
    \\    tensor (names below model.llm., e.g. layers.0.attn.wo_ud.weight)
    \\
;

const loop_hidden = 2304;
const loop_max_rows = 4096;
const ink2_layers = 24;
const embed_host_env = "TF_INK2_EMBED_HOST";
const dev_context = 2048;
const dev_chunk_rows = 64;
const device_limit: u64 = 30 << 30; // ASSUMPTION: the dev pod's GB10 pool shows about 31 GiB available
const stats_name = "weights.json";

const Args = struct {
    sub: []const u8,
    out: ?[]const u8 = null,
    model: ?[]const u8 = null,
    in: ?[]const u8 = null,
    layers: ?ink2.budget.LayerRange = null,
    readback: ?[]const u8 = null,
    rows: ?usize = null,
    layer: ?u32 = null,
    file: ?[]const u8 = null,
    mult: f32 = 1.0,
    pass: ?u32 = null,
    n: ?usize = null,
    pos0: usize = 0,
};

fn parseLayers(text: []const u8) !ink2.budget.LayerRange {
    const dash = std.mem.indexOfScalar(u8, text, '-') orelse return error.BadLayers;
    return .{ .first = try std.fmt.parseInt(u32, text[0..dash], 10), .last = try std.fmt.parseInt(u32, text[dash + 1 ..], 10) };
}

fn parseArgs(args: []const [:0]const u8) !Args {
    if (args.len < 2) return error.Usage;
    var a: Args = .{ .sub = args[1] };
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const v: []const u8 = args[i + 1];
        if (std.mem.eql(u8, args[i], "--out")) a.out = v else if (std.mem.eql(u8, args[i], "--model")) a.model = v else if (std.mem.eql(u8, args[i], "--in")) a.in = v else if (std.mem.eql(u8, args[i], "--layers")) a.layers = try parseLayers(v) else if (std.mem.eql(u8, args[i], "--readback")) a.readback = v else if (std.mem.eql(u8, args[i], "--rows")) a.rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--layer")) a.layer = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, args[i], "--file")) a.file = v else if (std.mem.eql(u8, args[i], "--mult")) a.mult = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, args[i], "--pass")) a.pass = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, args[i], "--pos0")) a.pos0 = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--n")) a.n = try std.fmt.parseInt(usize, v, 10) else return error.Usage;
    }
    return a;
}

pub const smoke_words = 1024;
pub const smoke_value: u32 = 0xC0FFEE;
const smoke_threads = 256;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const a = parseArgs(args) catch |e| {
        if (e == error.Usage) std.debug.print("{s}", .{usage});
        return e;
    };
    const out = a.out orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    if (std.mem.eql(u8, a.sub, "smoke")) {
        try smoke(init, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "weights")) {
        try weights(init, a, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "moe")) {
        const in = a.in orelse {
            std.debug.print("{s}", .{usage});
            return 2;
        };
        try moe(init, a, in, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "loop")) {
        const in = a.in orelse {
            std.debug.print("{s}", .{usage});
            return 2;
        };
        try loopLayer(init, a, in, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "ablit")) {
        try ablitRows(init, a, a.in orelse {
            std.debug.print("{s}", .{usage});
            return 2;
        }, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "attn")) {
        try attnTrace(init, a, a.in orelse {
            std.debug.print("{s}", .{usage});
            return 2;
        }, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "actq")) {
        try actq(init, a.in orelse {
            std.debug.print("{s}", .{usage});
            return 2;
        }, out);
        return 0;
    }
    std.debug.print("unknown subcommand: {s}\n{s}", .{ a.sub, usage });
    return 2;
}

fn rowDtype(d: ink2.weights.Dev) !ink2.check_rows.Dtype {
    return switch (d.dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .u8, .f8_e4m3 => .u8,
        else => error.RowDtype,
    };
}

fn weights(init: std.process.Init, a: Args, out: []const u8) !void {
    const model = a.model orelse {
        std.debug.print("{s}", .{usage});
        return error.Usage;
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();

    const embed_host = try environment.read(bool, embed_host_env, false);
    const o: ink2.budget.Options = .{ .context = dev_context, .streams = 1, .chunk_rows = dev_chunk_rows, .embed_host = embed_host, .layers = a.layers };
    var w = try ink2.weights.load(init.gpa, init.io, ink2.weights.deviceUploader(ctx.d), model, o, device_limit);
    defer w.deinit();
    const peak = ink2.proc_status.peakMib(init.gpa, init.io);

    var rows = try ink2.check_rows.Writer.init(init.gpa, init.io, out);
    defer rows.deinit();
    var names = std.mem.tokenizeScalar(u8, a.readback orelse "", ',');
    while (names.next()) |name| {
        const d = w.find(name) orelse {
            std.debug.print("no uploaded tensor {s}\n", .{name});
            return error.UnknownTensor;
        };
        const host = try init.gpa.alloc(u8, d.bytes);
        defer init.gpa.free(host);
        try cuda.memory.download(ctx.d, d.ptr, host);
        var shape: [4]usize = undefined;
        for (d.shape[0..d.rank], 0..) |n, i| shape[i] = @intCast(n);
        try rows.put(name, try rowDtype(d), shape[0..d.rank], host);
    }
    try rows.finish();

    const peak_mib = try peak;
    const stats = try std.fmt.allocPrint(init.gpa, "{{\"peak_rss_mib\":{d},\"device_bytes\":{d},\"budget_weights\":{d},\"budget_total\":{d},\"embed_host\":{s},\"layers\":[{d},{d}]}}\n", .{ peak_mib, w.device_bytes, w.plan.weights, w.plan.total, if (embed_host) "true" else "false", w.first, w.last });
    defer init.gpa.free(stats);
    try rows.dir.writeFile(init.io, .{ .sub_path = stats_name, .data = stats });
    std.debug.print("device_bytes {d} budget_weights {d} peak_rss_mib {d}\n", .{ w.device_bytes, w.plan.weights, peak_mib });
}

fn smoke(init: std.process.Init, out: []const u8) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try ink2.loop_kernels.Kernels.load(ctx.d);
    defer k.unload();

    var buf = try cuda.DeviceBuffer.alloc(ctx.d, smoke_words * @sizeOf(u32));
    defer buf.free();
    var args: cuda.Args = .{};
    args.add(@as(u64, buf.ptr));
    args.add(@as(i32, smoke_words));
    args.add(smoke_value);
    try cuda.launch.launch(k.fill, .{ .grid = .{ .x = smoke_words / smoke_threads }, .block = .{ .x = smoke_threads } }, stream, &args);
    try stream.synchronize();
    var host: [smoke_words]u32 = undefined;
    try buf.download(0, std.mem.sliceAsBytes(&host));

    var rows = try ink2.check_rows.Writer.init(init.gpa, init.io, out);
    defer rows.deinit();
    try rows.put("smoke", .u32, &.{smoke_words}, std.mem.sliceAsBytes(&host));
    try rows.finish();
}

const shapes_name = "shapes.json";
const actq_kinds = [_]struct { prefix: []const u8, out: []const u8, mx: bool }{
    .{ .prefix = "x_nvfp4_", .out = "nvfp4_", .mx = false },
    .{ .prefix = "x_mxfp8_", .out = "mxfp8_", .mx = true },
};

/// Every `x_*_<k>` entry of shapes.json is [rows, k]; the key names its input file.
fn actq(init: std.process.Init, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try ink2.loop_kernels.Kernels.load(ctx.d);
    defer k.unload();

    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const text = try in_dir.readFileAlloc(io, shapes_name, gpa, .limited(1 << 16));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.ArrayHashMap([2]usize), gpa, text, .{});
    defer parsed.deinit();

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    for (actq_kinds) |kind| {
        for (parsed.value.map.keys(), parsed.value.map.values()) |key, shape| {
            if (!std.mem.startsWith(u8, key, kind.prefix)) continue;
            const rows, const kdim = .{ shape[0], shape[1] };
            const file = try std.fmt.allocPrint(gpa, "{s}.bf16", .{key});
            defer gpa.free(file);
            const x_host = try in_dir.readFileAlloc(io, file, gpa, .limited(rows * kdim * 2 + 1));
            defer gpa.free(x_host);
            if (x_host.len != rows * kdim * 2) return error.ShapeMismatch;

            var x = try cuda.DeviceBuffer.alloc(ctx.d, x_host.len);
            defer x.free();
            var y = try cuda.DeviceBuffer.alloc(ctx.d, rows * kdim * 4);
            defer y.free();
            var values = try cuda.DeviceBuffer.alloc(ctx.d, rows * kdim * 4);
            defer values.free();
            // NVFP4: block scales float32 [rows, k/16]; MXFP8: exponents u8 [rows, k/32]
            const side_len = if (kind.mx) rows * (kdim / ink2.loop_kernels.mxfp8_block) else rows * (kdim / ink2.loop_kernels.nvfp4_block) * 4;
            var side = try cuda.DeviceBuffer.alloc(ctx.d, side_len);
            defer side.free();
            var amax = try cuda.DeviceBuffer.alloc(ctx.d, rows * 4);
            defer amax.free();
            try x.upload(0, x_host);
            if (kind.mx) {
                try ink2.loop_kernels.actMxfp8Codes(&k, stream, x.ptr, y.ptr, values.ptr, side.ptr, rows, kdim);
            } else {
                try ink2.loop_kernels.actNvfp4Codes(&k, stream, x.ptr, y.ptr, values.ptr, side.ptr, amax.ptr, rows, kdim);
            }
            try stream.synchronize();
            const suffix = key[kind.prefix.len..];
            try putDevice(gpa, &rows_out, &y, "{s}{s}", .{ kind.out, suffix }, .f32, &.{ rows, kdim });
            try putDevice(gpa, &rows_out, &values, "{s}values_{s}", .{ kind.out, suffix }, .f32, &.{ rows, kdim });
            if (kind.mx) {
                try putDevice(gpa, &rows_out, &side, "{s}exps_{s}", .{ kind.out, suffix }, .u8, &.{ rows, kdim / ink2.loop_kernels.mxfp8_block });
            } else {
                try putDevice(gpa, &rows_out, &side, "{s}scales_{s}", .{ kind.out, suffix }, .f32, &.{ rows, kdim / ink2.loop_kernels.nvfp4_block });
                try putDevice(gpa, &rows_out, &amax, "{s}amax_{s}", .{ kind.out, suffix }, .f32, &.{rows});
            }
        }
    }
    try rows_out.finish();
}

/// Downloads a device buffer whole and writes it as a row named by `fmt`.
fn putDevice(gpa: std.mem.Allocator, rows_out: *ink2.check_rows.Writer, buf: *cuda.DeviceBuffer, comptime fmt: []const u8, name_args: anytype, dtype: ink2.check_rows.Dtype, shape: []const usize) !void {
    var elems: usize = 1;
    for (shape) |n| elems *= n;
    const host = try gpa.alloc(u8, elems * dtype.size());
    defer gpa.free(host);
    try buf.download(0, host);
    const name = try std.fmt.allocPrint(gpa, fmt, name_args);
    defer gpa.free(name);
    try rows_out.put(name, dtype, shape, host);
}

const ablit_y_name = "y.bf16";

/// DIR/y.bf16 ([N, 2304]) through the edit file's layer `--layer`, scales times `--mult`; the result is written as y.
fn ablitRows(init: std.process.Init, a: Args, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const file_path = a.file orelse return error.Usage;
    const layer = a.layer orelse return error.Usage;
    const hidden = loop_hidden;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try ink2.loop_kernels.Kernels.load(ctx.d);
    defer k.unload();

    const edit = try ink2.ablit.load(gpa, io, file_path, hidden, ink2_layers);
    defer edit.deinit(gpa);
    var st = try ink2.ablit.upload(gpa, ctx.d, edit, a.mult);
    defer ink2.ablit.close(&st, gpa);

    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const y_host = try in_dir.readFileAlloc(io, ablit_y_name, gpa, .limited(loop_max_rows * hidden * 2 + 1));
    defer gpa.free(y_host);
    if (y_host.len == 0 or y_host.len % (hidden * 2) != 0) return error.ShapeMismatch;
    const rows = y_host.len / (hidden * 2);
    var y = try cuda.DeviceBuffer.fromHost(ctx.d, y_host);
    defer y.free();
    try ink2.ablit.apply(&k, &st, stream, layer, y.ptr, rows, hidden);
    try stream.synchronize();

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    try putDevice(gpa, &rows_out, &y, "y", .{}, .bf16, &.{ rows, hidden });
    try rows_out.finish();
}

const moe_x_name = "x.bf16";
const moe_max_rows = 16;

/// Layer L (`--layers L-L`) of the model on the rows of DIR/x.bf16; the layer's router picks and weights are written too.
fn moe(init: std.process.Init, a: Args, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const range = a.layers orelse return error.Usage;
    const rows = a.rows orelse return error.Usage;
    if (range.first != range.last) return error.Usage;
    if (rows == 0 or rows > moe_max_rows) return error.BadRows;
    const m = ink2.moe;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var loop = try ink2.loop_kernels.Kernels.load(ctx.d);
    defer loop.unload();
    var op = try ink2.ops.Kernels.load(ctx.d);
    defer op.unload();

    const embed_host = try environment.read(bool, embed_host_env, false);
    const o: ink2.budget.Options = .{ .context = dev_context, .streams = 1, .chunk_rows = dev_chunk_rows, .embed_host = embed_host, .layers = a.layers };
    var w = try ink2.weights.load(gpa, io, ink2.weights.deviceUploader(ctx.d), model, o, device_limit);
    defer w.deinit();

    const x_bytes = rows * m.hidden * 2;
    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const x_host = try in_dir.readFileAlloc(io, moe_x_name, gpa, .limited(x_bytes + 1));
    defer gpa.free(x_host);
    if (x_host.len != x_bytes) return error.ShapeMismatch;
    var x = try cuda.DeviceBuffer.fromHost(ctx.d, x_host);
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(ctx.d, x_bytes);
    defer y.free();
    var sc = try m.Scratch.init(ctx.d, rows);
    defer sc.deinit();

    const env: m.Env = .{ .driver = ctx.d, .loop = &loop, .ops = &op };
    try m.forward(&env, stream, &w.layers[range.first], w.prec[range.first], x.ptr, y.ptr, rows, &sc);
    try stream.synchronize();

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    try putDevice(gpa, &rows_out, &y, "out", .{}, .bf16, &.{ rows, m.hidden });
    try putDevice(gpa, &rows_out, &sc.pick, "pick", .{}, .i32, &.{ rows, m.top });
    try putDevice(gpa, &rows_out, &sc.gamma, "gamma", .{}, .f32, &.{ rows, m.top });
    try rows_out.finish();
}


/// DIR/name must hold exactly `bytes`; returns it on the device.
fn uploadFile(d: *const cuda.Driver, gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: usize) !cuda.DeviceBuffer {
    const host = try dir.readFileAlloc(io, name, gpa, .limited(bytes + 1));
    defer gpa.free(host);
    if (host.len != bytes) return error.ShapeMismatch;
    return cuda.DeviceBuffer.fromHost(d, host);
}

/// Core layer L (`--layer`) of the model on the files of DIR: messages of the bank, then each stream's controller with
/// DIR/norm.bf16 as the sublayer norm weight (the tool applies one norm to all four streams).
fn loopLayer(init: std.process.Init, a: Args, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const layer = a.layer orelse return error.Usage;
    const rows = a.rows orelse return error.Usage;
    if (rows == 0 or rows > loop_max_rows) return error.BadRows;
    const lk = ink2.loop_kernels;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var k = try lk.Kernels.load(ctx.d);
    defer k.unload();

    const embed_host = try environment.read(bool, embed_host_env, false);
    const o: ink2.budget.Options = .{ .context = dev_context, .streams = 1, .chunk_rows = dev_chunk_rows, .embed_host = embed_host, .layers = .{ .first = layer, .last = layer } };
    var w = try ink2.weights.load(gpa, io, ink2.weights.deviceUploader(ctx.d), model, o, device_limit);
    defer w.deinit();
    const core_start = (w.cfg.loop orelse return error.NotCoreLayer).core_start;
    if (layer < core_start) return error.NotCoreLayer;
    const ctl = w.layers[layer].controllers orelse return error.NotCoreLayer;
    const qa = w.cross_q orelse return error.NoCrossQueries;
    const qkv = w.cross_kv_q orelse return error.NoCrossQueries;

    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const block_bytes = rows * loop_hidden * 2;
    const f32_bytes = rows * loop_hidden * 4;
    var bank = try uploadFile(ctx.d, gpa, io, in_dir, "bank.bf16", 2 * block_bytes);
    defer bank.free();
    var dest = try uploadFile(ctx.d, gpa, io, in_dir, "dest.f32", f32_bytes);
    defer dest.free();
    var norm = try uploadFile(ctx.d, gpa, io, in_dir, "norm.bf16", loop_hidden * 2);
    defer norm.free();
    var msgs = try cuda.DeviceBuffer.alloc(ctx.d, lk.n_streams * f32_bytes);
    defer msgs.free();
    var h = try cuda.DeviceBuffer.alloc(ctx.d, rows * lk.gate_rank * 2);
    defer h.free();
    var y = try cuda.DeviceBuffer.alloc(ctx.d, block_bytes);
    defer y.free();

    try lk.messages(&k, stream, .{ bank.ptr, bank.ptr + block_bytes }, qa.ptr, qkv.ptr, layer - core_start, msgs.ptr, rows);
    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    try stream.synchronize();
    try putDevice(gpa, &rows_out, &msgs, "msgs", .{}, .f32, &.{ lk.n_streams, rows, loop_hidden });
    for (lk.stream_names, 0..) |name, s| {
        try lk.controller(&k, stream, .{ .dest = dest.ptr, .msg = msgs.ptr + s * f32_bytes, .down = ctl[s].down.ptr, .gain = ctl[s].gain.ptr, .forget = ctl[s].forget.ptr, .norm = norm.ptr, .h = h.ptr, .out = y.ptr, .rows = rows });
        try stream.synchronize();
        try putDevice(gpa, &rows_out, &y, "ctl_{s}", .{name}, .bf16, &.{ rows, loop_hidden });
    }
    try rows_out.finish();
}

const attn_xs_name = "xs.bf16";
const attn_inputs = 3;
const attn_max_rows = 512;
const attn_context_align = 64;

/// The steps of DIR/xs.bf16 through visit (`--layer`, `--pass`) of the model, `--rows` rows a call; the output of all
/// `--n` steps is written as out.
fn attnTrace(init: std.process.Init, a: Args, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const layer = a.layer orelse return error.Usage;
    const pass = a.pass orelse return error.Usage;
    const n = a.n orelse return error.Usage;
    const rows = a.rows orelse return error.Usage;
    if (n == 0 or rows == 0 or rows > attn_max_rows) return error.BadRows;
    const hidden = loop_hidden;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();

    const context = std.mem.alignForward(usize, @max(a.pos0 + n, dev_context), attn_context_align);
    const embed_host = try environment.read(bool, embed_host_env, false);
    const o: ink2.budget.Options = .{ .context = context, .streams = 1, .chunk_rows = @intCast(@max(rows, dev_chunk_rows)), .embed_host = embed_host, .layers = .{ .first = layer, .last = layer } };
    var w = try ink2.weights.load(gpa, io, ink2.weights.deviceUploader(ctx.d), model, o, device_limit);
    defer w.deinit();
    const layer_plan = try ink2.plan.build(&w.cfg.text, w.prec[0..w.cfg.text.layers]);
    const sched = try ink2.visits.build(&w.cfg, &layer_plan);
    const visit = for (sched.visits[0..sched.n]) |v| {
        if (v.layer == layer and v.pass == pass) break v;
    } else return error.NoSuchVisit;

    const co: ink2.cache.Options = .{ .context = context, .streams = 1, .chunk_rows = @intCast(rows) };
    var cache = try ink2.cache.Cache.init(ctx.d, &sched, &w.cfg.text, co);
    defer cache.deinit();
    var c: ink2.engine.Ctx = undefined;
    try c.init(ctx.d, &w, sched, &cache, co);
    defer c.deinit();

    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const row_bytes = hidden * 2;
    const xs = try in_dir.readFileAlloc(io, attn_xs_name, gpa, .limited(n * attn_inputs * row_bytes + 1));
    defer gpa.free(xs);
    if (xs.len != n * attn_inputs * row_bytes) return error.ShapeMismatch;

    if (a.pos0 > 0) try attnPrefix(gpa, io, in_dir, &cache, visit, a.pos0, &w.cfg.text);

    var x: [attn_inputs]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (x[0..made]) |*b| b.free();
    for (&x) |*b| {
        b.* = try cuda.DeviceBuffer.alloc(ctx.d, rows * row_bytes);
        made += 1;
    }
    var out_buf = try cuda.DeviceBuffer.alloc(ctx.d, n * row_bytes);
    defer out_buf.free();
    const stage = try gpa.alloc(u8, rows * row_bytes);
    defer gpa.free(stage);

    var at: usize = 0;
    while (at < n) : (at += rows) {
        const m = @min(rows, n - at);
        for (x, 0..) |b, j| {
            for (0..m) |r| @memcpy(stage[r * row_bytes ..][0..row_bytes], xs[((at + r) * attn_inputs + j) * row_bytes ..][0..row_bytes]);
            try b.upload(0, stage[0 .. m * row_bytes]);
        }
        try ink2.attn_visit.run(&c, stream, visit, &w.layers[layer], .{ x[0].ptr, x[1].ptr, x[2].ptr }, m, a.pos0 + at, 0, out_buf.ptr + at * row_bytes);
        try stream.synchronize();
    }

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    try putDevice(gpa, &rows_out, &out_buf, "out", .{}, .bf16, &.{ n, hidden });
    try rows_out.finish();
}

const attn_k0_name = "k0.bf16";
const attn_v0_name = "v0.bf16";

/// Writes the rows before `pos0` (k0.bf16, v0.bf16: bf16 [keep, kv_heads, 128], the last `keep` positions) into the visit's rings.
fn attnPrefix(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, cache: *ink2.cache.Cache, visit: ink2.visits.Visit, pos0: usize, t: *const ink2.config.Text) !void {
    const kv_heads: usize = t.heads(visit.layer).kv;
    const layer = cache.layerAt(visit.kv);
    const slots = cache.kvAt(visit.kv, 0);
    const head_row = ink2.kernels.head_dim * 2;
    const limit = pos0 * kv_heads * head_row;
    for ([_]struct { name: []const u8, at: u64 }{ .{ .name = attn_k0_name, .at = slots.k }, .{ .name = attn_v0_name, .at = slots.v } }) |f| {
        const src = try dir.readFileAlloc(io, f.name, gpa, .limited(limit + 1));
        defer gpa.free(src);
        if (src.len == 0 or src.len % (kv_heads * head_row) != 0) return error.ShapeMismatch;
        const keep = src.len / (kv_heads * head_row);
        if (keep > pos0 or keep > layer.ring) return error.ShapeMismatch;
        const ring: usize = layer.ring;
        const staging = try gpa.alloc(u8, ring * head_row);
        defer gpa.free(staging);
        for (0..kv_heads) |h| {
            ink2.cache.headRing(staging, ring, kv_heads, h, pos0 - keep, keep, src);
            const view: cuda.DeviceBuffer = .{ .d = cache.d, .ptr = f.at + h * ring * head_row, .len = staging.len, .borrowed = true };
            try view.upload(0, staging);
        }
    }
}
