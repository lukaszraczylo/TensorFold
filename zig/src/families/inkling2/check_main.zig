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
    \\decode --model DIR --steps FILE.json --visits A-B [--teacher GOLDEN_DIR] [--layers staged] [--dump-kv L:P,...] --out DIR: runs the single-token steps of FILE.json ({"steps": [[id], ...]}) from position 0 through visits 0..B (B at most 47), the carried input after each visit, each visit's MoE input and a pass > 0 visit's messages replaced by GOLDEN_DIR's (ref.capture layout: visit<VV>.f32, mlp<VV>.f32, msg<VV>.f32) when given, and writes step<s>/visit<VV>.f32 and step<s>/mlp<VV>.f32 (the MoE input before the replacement), for a visit with pass > 0 step<s>/xs<VV>.f32 and msg<VV>.f32 (controller outputs, messages), for visits A..B, and a manifest.json in the reference's layout; --dump-kv writes the K and V rows of those (layer, pass) visits after the last step as kv/L<l>p<p>.<k|v>.bf16 ([steps, 8, 128])
    \\    With --visits ending at the last visit it also runs the head on the carried input after it (the golden's when --teacher is given, so the logits show the head alone) and writes step<s>/logits.f32 ([unpadded_vocab], softcapped fp32) and step<s>/argmax.u32 (the device pick, checked against a host pick); with --teacher also step<s>/logits_free.f32, the logits of the engine's own last-visit output
    \\greedy --model DIR --prompts FILE.json --n N [--chunk-rows R] --out DIR: FILE.json is {"prompts": [[id, ...], ...]}; each prompt is prefilled in chunks of R rows (default 512) from position 0 (conv states cleared between prompts), then N tokens are picked greedily (lowest id on ties); writes tokens.u32 ([prompts, N]) and p<P>/logits<GG>.f32 ([unpadded_vocab]) for each picked token
    \\prefill --model DIR --ids FILE.json [--chunk-rows R] [--decode N] [--teacher GOLDEN_DIR] [--dump-kv L:P,...] --out DIR: FILE.json is {"steps": [[id, ...], [id], ...]}; step 0 is prefilled in chunks of R rows (default 512) from position 0, then steps 1..N run as single-token decode steps (teacher-forced ids); writes ref.capture's all-rows layout (step<s>/visit<VV>.f32 [rows, qn, 2304], mlp<VV>.f32, xs<VV>.f32, msg<VV>.f32), step<s>/logits.f32 and argmax.u32 (the last row's), and a manifest.json; with GOLDEN_DIR (captured with --all-rows) the carried input, the MoE input and the messages of every row are replaced by the reference's; --dump-kv writes the K and V rows of the given (layer, pass) visits for all positions
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
const staged_layers = "staged";

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
    steps: ?[]const u8 = null,
    teacher: ?[]const u8 = null,
    visits: ?ink2.budget.LayerRange = null,
    staged: bool = false,
    dump_kv: ?[]const u8 = null,
    prompts: ?[]const u8 = null,
    ids: ?[]const u8 = null,
    chunk_rows: u32 = ink2.prefill.default_chunk_rows,
    decode: usize = 0,
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
        if (std.mem.eql(u8, args[i], "--out")) a.out = v else if (std.mem.eql(u8, args[i], "--model")) a.model = v else if (std.mem.eql(u8, args[i], "--in")) a.in = v else if (std.mem.eql(u8, args[i], "--layers")) {
            if (std.mem.eql(u8, v, staged_layers)) a.staged = true else a.layers = try parseLayers(v);
        } else if (std.mem.eql(u8, args[i], "--steps")) a.steps = v else if (std.mem.eql(u8, args[i], "--teacher")) a.teacher = v else if (std.mem.eql(u8, args[i], "--visits")) a.visits = try parseLayers(v) else if (std.mem.eql(u8, args[i], "--dump-kv")) a.dump_kv = v else if (std.mem.eql(u8, args[i], "--prompts")) a.prompts = v else if (std.mem.eql(u8, args[i], "--ids")) a.ids = v else if (std.mem.eql(u8, args[i], "--chunk-rows")) a.chunk_rows = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, args[i], "--decode")) a.decode = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--readback")) a.readback = v else if (std.mem.eql(u8, args[i], "--rows")) a.rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--layer")) a.layer = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, args[i], "--file")) a.file = v else if (std.mem.eql(u8, args[i], "--mult")) a.mult = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, args[i], "--pass")) a.pass = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, args[i], "--pos0")) a.pos0 = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, args[i], "--n")) a.n = try std.fmt.parseInt(usize, v, 10) else return error.Usage;
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
    if (std.mem.eql(u8, a.sub, "decode")) {
        try decodeRun(init, a, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "greedy")) {
        try greedyRun(init, a, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "prefill")) {
        try prefillRun(init, a, out);
        return 0;
    }
    if (std.mem.eql(u8, a.sub, "conv")) {
        try convRun(init, a.in orelse {
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

const conv_width = 2304;

/// DIR's x, w, base and resid through tf_ink2_conv's static modes 1 and 2, `y_f32` 0 and 1, each from a zero state.
fn convRun(init: std.process.Init, in: []const u8, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var op = try ink2.ops.Kernels.load(ctx.d);
    defer op.unload();

    const in_dir = try std.Io.Dir.cwd().openDir(io, in, .{});
    defer in_dir.close(io);
    const x_host = try in_dir.readFileAlloc(io, "x.bf16", gpa, .limited(1 << 26));
    defer gpa.free(x_host);
    const rows = x_host.len / (conv_width * 2);
    if (rows == 0 or x_host.len != rows * conv_width * 2) return error.ShapeMismatch;
    const w_host = try in_dir.readFileAlloc(io, "w.bf16", gpa, .limited(conv_width * 8 + 1));
    defer gpa.free(w_host);
    const base_host = try in_dir.readFileAlloc(io, "base.bf16", gpa, .limited(x_host.len + 1));
    defer gpa.free(base_host);
    const resid_host = try in_dir.readFileAlloc(io, "resid.f32", gpa, .limited(rows * conv_width * 4 + 1));
    defer gpa.free(resid_host);
    if (w_host.len != conv_width * 8 or base_host.len != x_host.len or resid_host.len != rows * conv_width * 4) return error.ShapeMismatch;

    var x = try cuda.DeviceBuffer.alloc(ctx.d, x_host.len);
    defer x.free();
    var w = try cuda.DeviceBuffer.alloc(ctx.d, w_host.len);
    defer w.free();
    var base = try cuda.DeviceBuffer.alloc(ctx.d, base_host.len);
    defer base.free();
    var resid = try cuda.DeviceBuffer.alloc(ctx.d, resid_host.len);
    defer resid.free();
    var state = try cuda.DeviceBuffer.alloc(ctx.d, 3 * conv_width * 4);
    defer state.free();
    var out_bf = try cuda.DeviceBuffer.alloc(ctx.d, x_host.len);
    defer out_bf.free();
    try x.upload(0, x_host);
    try w.upload(0, w_host);
    try base.upload(0, base_host);

    const zeros = try gpa.alloc(u8, 3 * conv_width * 4);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    for ([_]i32{ 0, 1 }) |f| {
        for ([_]i32{ 1, 2 }) |mode| {
            try state.upload(0, zeros);
            if (mode == 2) try resid.upload(0, resid_host);
            const args: ink2.ops.ConvArgs = .{
                .x = x.ptr,
                .state = state.ptr,
                .w = w.ptr,
                .base = if (mode == 1) base.ptr else 0,
                .resid = resid.ptr,
                .out = out_bf.ptr,
                .x_bf16 = 1,
                .c = conv_width,
                .rows = @intCast(rows),
                .mode = mode,
                .y_f32 = f,
            };
            try ink2.ops.conv(&op, stream, args, 1);
            try stream.synchronize();
            if (mode == 1) try putDevice(gpa, &rows_out, &resid, "m1_y{d}", .{f}, .f32, &.{ rows, conv_width }) else try putDevice(gpa, &rows_out, &out_bf, "m2_y{d}", .{f}, .bf16, &.{ rows, conv_width });
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

const decode_chunk_rows = 1;
const f32_size = @sizeOf(f32);
/// Rows of a visit's carried vector, MoE input, controller outputs and messages: the most any of them has.
const max_row_kinds = ink2.loop_kernels.n_streams;

const DecodeSteps = struct { steps: []const []const u32 };

/// Rows `first_row`.. of the all-rows file `path` (qn vectors a row); a later chunk of the same step adds to the file.
fn writeRows(dir: std.Io.Dir, io: std.Io, path: []const u8, qn: usize, first_row: u64, data: []const f32) !void {
    if (std.fs.path.dirname(path)) |parent| try dir.createDirPath(io, parent);
    const file = try dir.createFile(io, path, .{ .truncate = false });
    defer file.close(io);
    try file.writePositionalAll(io, std.mem.sliceAsBytes(data), first_row * qn * loop_hidden * f32_size);
}

fn readRows(dir: std.Io.Dir, io: std.Io, path: []const u8, qn: usize, first_row: u64, out: []f32) !void {
    const file = try dir.openFile(io, path, .{});
    defer file.close(io);
    const got = try file.readPositionalAll(io, std.mem.sliceAsBytes(out), first_row * qn * loop_hidden * f32_size);
    if (got != out.len * f32_size) return error.ShapeMismatch;
}

/// Hands decode.forward the golden's carried input, MoE input and messages of the rows of the step being run.
const GoldenTeacher = struct {
    io: std.Io,
    dir: std.Io.Dir,
    sched: *const ink2.visits.Schedule,
    step: usize = 0,
    step_pos0: u64 = 0,
    raw: []f32,
    buf: []f32,
    err: ?anyerror = null,

    fn input(ctx: *anyopaque, v: u32, span: ink2.decode.Span) ?[]const f32 {
        const t: *GoldenTeacher = @ptrCast(@alignCast(ctx));
        return t.read("visit", v, span, ink2.decode.carriedRows(t.sched, v));
    }

    fn mlp(ctx: *anyopaque, v: u32, span: ink2.decode.Span) ?[]const f32 {
        const t: *GoldenTeacher = @ptrCast(@alignCast(ctx));
        return t.read("mlp", v, span, 1);
    }

    fn msg(ctx: *anyopaque, v: u32, span: ink2.decode.Span) ?[]const f32 {
        const t: *GoldenTeacher = @ptrCast(@alignCast(ctx));
        return t.read("msg", v, span, ink2.loop_kernels.n_streams);
    }

    fn read(t: *GoldenTeacher, kind: []const u8, v: u32, span: ink2.decode.Span, qn: usize) ?[]const f32 {
        return t.load(kind, v, span, qn) catch |e| {
            t.err = e;
            return null;
        };
    }

    fn load(t: *GoldenTeacher, kind: []const u8, v: u32, span: ink2.decode.Span, qn: usize) ![]const f32 {
        var name: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "step{d}/{s}{d:0>2}.f32", .{ t.step, kind, v });
        const n = qn * span.rows * loop_hidden;
        try readRows(t.dir, t.io, path, qn, span.p0 - t.step_pos0, t.raw[0..n]);
        ink2.check_rows.toDeviceLayout(t.raw[0..n], t.buf[0..n], qn, span.rows, loop_hidden);
        return t.buf[0..n];
    }
};

/// Writes the rows of each visit from `first` on in the reference's all-rows layout: step<s>/<kind><VV>.f32.
const DecodeTrace = struct {
    d: *const cuda.Driver,
    io: std.Io,
    dir: std.Io.Dir,
    rows_out: *ink2.check_rows.Writer,
    first: u32,
    step: usize = 0,
    step_pos0: u64 = 0,
    raw: []f32,
    buf: []f32,
    /// Set with a teacher: the engine's own head input of the last visit goes through the head before the teacher's replaces it.
    head_free: ?*HeadRun = null,

    fn put(t: *DecodeTrace, kind: []const u8, v: u32, span: ink2.decode.Span, ptr: u64, qn: usize) !void {
        if (v < t.first) return;
        const n = qn * span.rows * loop_hidden;
        try cuda.memory.download(t.d, ptr, std.mem.sliceAsBytes(t.raw[0..n]));
        ink2.check_rows.toFileLayout(t.raw[0..n], t.buf[0..n], qn, span.rows, loop_hidden);
        var name: [64]u8 = undefined;
        try writeRows(t.dir, t.io, try std.fmt.bufPrint(&name, "step{d}/{s}{d:0>2}.f32", .{ t.step, kind, v }), qn, span.p0 - t.step_pos0, t.buf[0..n]);
    }

    fn visit(ctx: *anyopaque, v: u32, span: ink2.decode.Span, inp: u64, qn: u32) anyerror!void {
        const t: *DecodeTrace = @ptrCast(@alignCast(ctx));
        try put(t, "visit", v, span, inp, qn);
        if (t.head_free) |h| if (v == h.last_visit) {
            _ = try h.run(span.rows - 1);
            var name: [64]u8 = undefined;
            try h.put(t.rows_out, try std.fmt.bufPrint(&name, "step{d}/logits_free", .{t.step}));
        };
    }

    fn mlp(ctx: *anyopaque, v: u32, span: ink2.decode.Span, mlp_in: u64) anyerror!void {
        try put(@ptrCast(@alignCast(ctx)), "mlp", v, span, mlp_in, 1);
    }

    fn xs(ctx: *anyopaque, v: u32, span: ink2.decode.Span, ptr: u64) anyerror!void {
        try put(@ptrCast(@alignCast(ctx)), "xs", v, span, ptr, ink2.engine.max_inputs);
    }

    fn msg(ctx: *anyopaque, v: u32, span: ink2.decode.Span, ptr: u64) anyerror!void {
        try put(@ptrCast(@alignCast(ctx)), "msg", v, span, ptr, ink2.loop_kernels.n_streams);
    }
};

const config_name = "config.json";
const config_max_bytes = 1 << 20;
const max_kv_dumps = 8;
const KvVisit = struct { layer: u32, pass: u32 };

/// `6:0,6:1` -> (layer, pass) visits.
fn parseKvVisits(text: []const u8, out: *[max_kv_dumps]KvVisit) ![]const KvVisit {
    var n: usize = 0;
    var items = std.mem.splitScalar(u8, text, ',');
    while (items.next()) |item| {
        const colon = std.mem.indexOfScalar(u8, item, ':') orelse return error.Usage;
        if (n == max_kv_dumps) return error.Usage;
        out[n] = .{ .layer = try std.fmt.parseInt(u32, item[0..colon], 10), .pass = try std.fmt.parseInt(u32, item[colon + 1 ..], 10) };
        n += 1;
    }
    return out[0..n];
}

/// The schedule of the model at `dir` before its weights load: the layer kinds do not depend on the expert precision.
fn scheduleOf(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !ink2.visits.Schedule {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const path = try std.fs.path.join(arena.allocator(), &.{ dir, config_name });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena.allocator(), .limited(config_max_bytes));
    var why: ink2.config.Why = .{};
    const cfg = ink2.config.parse(arena.allocator(), text, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ path, why.text() });
        return e;
    };
    const prec: [ink2.config.max_layers]ink2.plan.Precision = @splat(.nvfp4);
    const p = try ink2.plan.build(&cfg.text, prec[0..cfg.text.layers]);
    return ink2.visits.build(&cfg, &p);
}

/// The K and V rows of each visit in `kv` after `steps` single-token steps from position 0, as kv/L<l>p<p>.<k|v>.bf16
/// ([steps, kv_heads, head_dim], ref.capture's layout); the cache holds a head's rows at slot pos % ring.
fn dumpKv(gpa: std.mem.Allocator, rows_out: *ink2.check_rows.Writer, cache: *const ink2.cache.Cache, sched: *const ink2.visits.Schedule, t: *const ink2.config.Text, kv: []const KvVisit, last_visit: u32, steps: usize) !void {
    const head_row = ink2.kernels.head_dim * 2;
    for (kv) |want| {
        const i = for (sched.visits[0..sched.n], 0..) |v, i| {
            if (v.layer == want.layer and v.pass == want.pass) break i;
        } else return error.KvVisitUnknown;
        if (i > last_visit) return error.KvVisitNotRun;
        const visit = sched.visits[i];
        const layer = cache.layerAt(visit.kv);
        const ring: usize = layer.ring;
        if (steps > ring) return error.KvRingWrapped;
        const kv_heads: usize = t.heads(visit.layer).kv;
        const slots = cache.kvAt(visit.kv, 0);
        const staging = try gpa.alloc(u8, kv_heads * ring * head_row);
        defer gpa.free(staging);
        const out = try gpa.alloc(u8, steps * kv_heads * head_row);
        defer gpa.free(out);
        for ([_]struct { kind: []const u8, at: u64 }{ .{ .kind = "k", .at = slots.k }, .{ .kind = "v", .at = slots.v } }) |f| {
            try cuda.memory.download(cache.d, f.at, staging);
            for (0..steps) |pos| for (0..kv_heads) |h| {
                @memcpy(out[(pos * kv_heads + h) * head_row ..][0..head_row], staging[(h * ring + pos % ring) * head_row ..][0..head_row]);
            };
            var name: [64]u8 = undefined;
            const file = try std.fmt.bufPrint(&name, "kv/L{d}p{d}.{s}", .{ want.layer, want.pass, f.kind });
            try rows_out.put(file, .bf16, &.{ steps, kv_heads, ink2.kernels.head_dim }, out);
        }
    }
}

/// The head on the carried input of the last visit: logits and the device pick, with the logits on the host.
const HeadRun = struct {
    c: *ink2.engine.Ctx,
    s: cuda.Stream,
    scratch: ink2.head.Scratch,
    host: []f32,
    last_visit: u32,

    fn init(gpa: std.mem.Allocator, c: *ink2.engine.Ctx, s: cuda.Stream) !HeadRun {
        const n: usize = c.text.unpadded_vocab;
        const host = try gpa.alloc(f32, n);
        errdefer gpa.free(host);
        return .{ .c = c, .s = s, .scratch = try ink2.head.Scratch.init(c.d, 1, n), .host = host, .last_visit = c.sched.n - 1 };
    }

    fn deinit(h: *HeadRun, gpa: std.mem.Allocator) void {
        h.scratch.deinit();
        gpa.free(h.host);
    }

    /// Logits of row `row` of the carried input now in the flow, into `host`; returns the device pick, which a host pick must repeat.
    fn run(h: *HeadRun, row: usize) !u32 {
        const at = h.c.flow.r.inp + row * h.c.text.hidden * @sizeOf(u16);
        try ink2.head.logits(h.c, h.s, at, 1, h.scratch.r.logits);
        return h.finish();
    }

    /// The pick and the host logits of the logits already in the scratch (`prefill.run` leaves its last row's there).
    fn finish(h: *HeadRun) !u32 {
        const r = h.scratch.r;
        try ink2.head.argmax(h.c, h.s, r.logits, 1, r.val, r.id);
        try h.s.synchronize();
        try cuda.memory.download(h.c.d, r.logits, std.mem.sliceAsBytes(h.host));
        var id: u32 = undefined;
        try cuda.memory.download(h.c.d, r.id, std.mem.asBytes(&id));
        if (id >= h.host.len) return error.Inkling2PickMismatch; // an all-NaN row picks (-inf, base)
        if (ink2.head.hostArgmax(h.host)) |want| if (want != id) {
            std.debug.print("head: device pick {d} but host pick {d}\n", .{ id, want });
            return error.Inkling2PickMismatch;
        };
        return id;
    }

    fn put(h: *const HeadRun, rows_out: *ink2.check_rows.Writer, name: []const u8) !void {
        try rows_out.put(name, .f32, &.{h.host.len}, std.mem.sliceAsBytes(h.host));
    }
};

/// The device, the weights through `last_visit`, the caches and the engine context, in place: `c` points into them.
const Session = struct {
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    w: ink2.weights.Weights,
    sched: ink2.visits.Schedule,
    cache: ink2.cache.Cache,
    c: ink2.engine.Ctx,

    /// Weights of the layers visits 0..`last_visit` (default: the whole schedule) read, or `layers` when given.
    fn open(s: *Session, gpa: std.mem.Allocator, io: std.Io, model: []const u8, last_visit: ?u32, layers: ?ink2.budget.LayerRange, chunk_rows: u32) !void {
        s.driver = try cuda.Driver.open();
        errdefer s.driver.close();
        s.ctx = try cuda.Context.init(&s.driver, 0);
        errdefer s.ctx.deinit();
        s.stream = try cuda.Stream.init(s.ctx.d, true);
        errdefer s.stream.deinit();

        const embed_host = try environment.read(bool, embed_host_env, false);
        const planned = try scheduleOf(gpa, io, model);
        const last = last_visit orelse planned.n - 1;
        if (last >= planned.n) return error.VisitsPastSchedule;
        const range = layers orelse ink2.budget.LayerRange{ .first = 0, .last = ink2.decode.layersRead(&planned, last) };
        const o: ink2.budget.Options = .{ .context = dev_context, .streams = 1, .chunk_rows = chunk_rows, .embed_host = embed_host, .layers = range };
        s.w = try ink2.weights.load(gpa, io, ink2.weights.deviceUploader(s.ctx.d), model, o, device_limit);
        errdefer s.w.deinit();
        const layer_plan = try ink2.plan.build(&s.w.cfg.text, s.w.prec[0..s.w.cfg.text.layers]);
        s.sched = try ink2.visits.build(&s.w.cfg, &layer_plan);
        if (s.sched.n != planned.n) return error.VisitsPastSchedule;

        const co: ink2.cache.Options = .{ .context = dev_context, .streams = 1, .chunk_rows = chunk_rows };
        s.cache = try ink2.cache.Cache.init(s.ctx.d, &s.sched, &s.w.cfg.text, co);
        errdefer s.cache.deinit();
        try s.c.init(s.ctx.d, &s.w, s.sched, &s.cache, co);
    }

    fn deinit(s: *Session) void {
        s.c.deinit();
        s.cache.deinit();
        s.w.deinit();
        s.stream.deinit();
        s.ctx.deinit();
        s.driver.close();
    }
};

/// Single-token steps of `--steps` from position 0 through visits 0..B of the model; see `usage`.
fn decodeRun(init: std.process.Init, a: Args, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const steps_path = a.steps orelse return error.Usage;
    const range = a.visits orelse return error.Usage;
    if (range.first > range.last) return error.Usage;

    const text = try std.Io.Dir.cwd().readFileAlloc(io, steps_path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(DecodeSteps, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const steps = parsed.value.steps;
    if (steps.len == 0 or steps.len > dev_context) return error.BadSteps;
    for (steps) |ids| if (ids.len != 1) return error.BadSteps;

    var kv_buf: [max_kv_dumps]KvVisit = undefined;
    const kv: []const KvVisit = if (a.dump_kv) |list| try parseKvVisits(list, &kv_buf) else &.{};

    var session: Session = undefined;
    try session.open(gpa, io, model, range.last, a.layers, decode_chunk_rows);
    defer session.deinit();
    const c = &session.c;
    const sched = session.sched;

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    const row_buf_len = max_row_kinds * decode_chunk_rows * loop_hidden;
    const trace_raw = try gpa.alloc(f32, row_buf_len);
    defer gpa.free(trace_raw);
    const trace_buf = try gpa.alloc(f32, row_buf_len);
    defer gpa.free(trace_buf);
    var tracer: DecodeTrace = .{ .d = c.d, .io = io, .dir = rows_out.dir, .rows_out = &rows_out, .first = range.first, .raw = trace_raw, .buf = trace_buf };

    var golden_dir: ?std.Io.Dir = null;
    defer if (golden_dir) |*dir| dir.close(io);
    const teacher_raw = try gpa.alloc(f32, row_buf_len);
    defer gpa.free(teacher_raw);
    const teacher_buf = try gpa.alloc(f32, row_buf_len);
    defer gpa.free(teacher_buf);
    var teacher: GoldenTeacher = .{ .io = io, .dir = undefined, .sched = &sched, .raw = teacher_raw, .buf = teacher_buf };
    if (a.teacher) |path| {
        golden_dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        teacher.dir = golden_dir.?;
    }

    // the head runs when the visits reach the last one
    const with_head = range.last + 1 == sched.n;
    var head_run: HeadRun = undefined;
    if (with_head) head_run = try HeadRun.init(gpa, c, session.stream);
    defer if (with_head) head_run.deinit(gpa);
    if (with_head and a.teacher != null) tracer.head_free = &head_run;

    const hooks: ink2.decode.Hooks = .{
        .trace = .{ .ctx = &tracer, .visit = DecodeTrace.visit, .mlp = DecodeTrace.mlp, .xs = DecodeTrace.xs, .msg = DecodeTrace.msg },
        .teacher = if (a.teacher != null) .{ .ctx = &teacher, .input = GoldenTeacher.input, .mlp = GoldenTeacher.mlp, .msg = GoldenTeacher.msg } else null,
    };
    const metas = try gpa.alloc(ink2.check_rows.StepMeta, steps.len);
    defer gpa.free(metas);
    for (steps, 0..) |ids, s| {
        tracer.step = s;
        tracer.step_pos0 = s;
        teacher.step = s;
        teacher.step_pos0 = s;
        metas[s] = .{ .n = 1, .pos0 = s };
        try ink2.decode.step(c, session.stream, 0, ids[0], s, range.last, hooks);
        if (teacher.err) |e| return e;
        if (with_head) {
            const pick = try head_run.run(0);
            var name: [64]u8 = undefined;
            try head_run.put(&rows_out, try std.fmt.bufPrint(&name, "step{d}/logits", .{s}));
            try rows_out.put(try std.fmt.bufPrint(&name, "step{d}/argmax", .{s}), .u32, &.{1}, std.mem.asBytes(&pick));
        }
    }

    try dumpKv(gpa, &rows_out, &session.cache, &sched, &session.w.cfg.text, kv, range.last, steps.len);
    var qn: [ink2.visits.max_visits]u32 = undefined;
    var loop_visits: [ink2.visits.max_visits]u32 = undefined;
    var n_loop: usize = 0;
    for (qn[0..sched.n], 0..) |*q, i| {
        q.* = ink2.decode.carriedRows(&sched, @intCast(i));
        if (sched.visits[i].pass > 0) {
            loop_visits[n_loop] = @intCast(i);
            n_loop += 1;
        }
    }
    try rows_out.finishRun(.{ .hidden = loop_hidden, .visits = sched.n, .unpadded_vocab = session.w.cfg.text.unpadded_vocab, .qn = qn[0..sched.n], .loop_visits = loop_visits[0..n_loop], .steps = metas });
    std.debug.print("decode: {d} steps, visits {d}-{d}, device_bytes {d}\n", .{ steps.len, range.first, range.last, session.w.device_bytes });
    if (with_head) {
        const b = ink2.head.deviceBytes(&session.w.cfg.text, 1);
        std.debug.print("head: unembed {d} B (in device_bytes), scratch {d} B\n", .{ b.unembed, b.scratch });
    }
}

const GreedyPrompts = struct { prompts: []const []const u32 };

/// Each prompt of `--prompts` prefilled in chunks, then `--n` greedy tokens; see `usage`.
fn greedyRun(init: std.process.Init, a: Args, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const prompts_path = a.prompts orelse return error.Usage;
    const n = a.n orelse return error.Usage;

    const text = try std.Io.Dir.cwd().readFileAlloc(io, prompts_path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(GreedyPrompts, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const prompts = parsed.value.prompts;
    if (prompts.len == 0 or n == 0) return error.BadPrompts;

    var session: Session = undefined;
    try session.open(gpa, io, model, null, a.layers, a.chunk_rows);
    defer session.deinit();
    const c = &session.c;
    const vocab = c.text.unpadded_vocab;
    for (prompts) |ids| {
        if (ids.len == 0 or ids.len + n > dev_context) return error.BadPrompts;
        for (ids) |id| if (id >= vocab) return error.BadPrompts;
    }

    var head_run = try HeadRun.init(gpa, c, session.stream);
    defer head_run.deinit(gpa);
    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    const tokens = try gpa.alloc(u32, prompts.len * n);
    defer gpa.free(tokens);

    const last = session.sched.n - 1;
    for (prompts, 0..) |ids, p| {
        try session.cache.clear(session.stream, 0);
        try ink2.prefill.run(c, session.stream, 0, ids, a.chunk_rows, head_run.scratch.r.logits, .{});
        var pick = try head_run.finish();
        var g: usize = 0;
        while (true) {
            tokens[p * n + g] = pick;
            var name: [64]u8 = undefined;
            try head_run.put(&rows_out, try std.fmt.bufPrint(&name, "p{d}/logits{d:0>2}", .{ p, g }));
            if (g + 1 == n) break;
            try ink2.decode.step(c, session.stream, 0, pick, ids.len + g, last, .{});
            pick = try head_run.run(0);
            g += 1;
        }
    }
    try rows_out.put("tokens", .u32, &.{ prompts.len, n }, std.mem.sliceAsBytes(tokens));
    try rows_out.finish();
    const b = ink2.head.deviceBytes(&session.w.cfg.text, 1);
    std.debug.print("greedy: {d} prompts, {d} tokens each, chunk rows {d}, device_bytes {d}, head unembed {d} B (in device_bytes), scratch {d} B\n", .{ prompts.len, n, a.chunk_rows, session.w.device_bytes, b.unembed, b.scratch });
}

/// Device bytes of everything a session holds beyond the weights: caches, attention, flow, MoE and head scratch.
fn sessionBytes(session: *const Session, head_scratch: u64) u64 {
    const c = &session.c;
    var total: u64 = session.cache.kv.len + session.cache.conv.len + c.scratch.mem.len + c.flow.mem.len + head_scratch;
    inline for (.{ "scores", "pick", "gamma", "act", "gu", "y", "yq", "dn" }) |f| total += @field(c.moe_sc, f).len;
    return total;
}

/// Step 0 of `--ids` prefilled in chunks, then `--decode` single-token steps; see `usage`.
fn prefillRun(init: std.process.Init, a: Args, out: []const u8) !void {
    const gpa = init.gpa;
    const io = init.io;
    const model = a.model orelse return error.Usage;
    const ids_path = a.ids orelse return error.Usage;

    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(DecodeSteps, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const steps = parsed.value.steps;
    if (steps.len == 0 or a.decode + 1 > steps.len or a.chunk_rows == 0) return error.BadSteps;
    var positions: usize = 0;
    for (steps[0 .. a.decode + 1], 0..) |ids, s| {
        if (ids.len == 0 or (s > 0 and ids.len != 1)) return error.BadSteps;
        positions += ids.len;
    }
    if (positions > dev_context) return error.BadSteps;

    var kv_buf: [max_kv_dumps]KvVisit = undefined;
    const kv: []const KvVisit = if (a.dump_kv) |list| try parseKvVisits(list, &kv_buf) else &.{};

    var session: Session = undefined;
    try session.open(gpa, io, model, null, a.layers, a.chunk_rows);
    defer session.deinit();
    const c = &session.c;
    const sched = session.sched;
    const vocab = c.text.unpadded_vocab;
    for (steps[0 .. a.decode + 1]) |ids| for (ids) |id| if (id >= vocab) return error.BadSteps;

    var rows_out = try ink2.check_rows.Writer.init(gpa, io, out);
    defer rows_out.deinit();
    const row_buf_len = max_row_kinds * @max(a.chunk_rows, 1) * loop_hidden;
    const bufs = try gpa.alloc(f32, 4 * row_buf_len);
    defer gpa.free(bufs);
    var tracer: DecodeTrace = .{ .d = c.d, .io = io, .dir = rows_out.dir, .rows_out = &rows_out, .first = 0, .raw = bufs[0..row_buf_len], .buf = bufs[row_buf_len .. 2 * row_buf_len] };
    var golden_dir: ?std.Io.Dir = null;
    defer if (golden_dir) |*dir| dir.close(io);
    var teacher: GoldenTeacher = .{ .io = io, .dir = undefined, .sched = &sched, .raw = bufs[2 * row_buf_len .. 3 * row_buf_len], .buf = bufs[3 * row_buf_len ..] };
    if (a.teacher) |path| {
        golden_dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        teacher.dir = golden_dir.?;
    }
    var head_run = try HeadRun.init(gpa, c, session.stream);
    defer head_run.deinit(gpa);
    if (a.teacher != null) tracer.head_free = &head_run;
    const hooks: ink2.decode.Hooks = .{
        .trace = .{ .ctx = &tracer, .visit = DecodeTrace.visit, .mlp = DecodeTrace.mlp, .xs = DecodeTrace.xs, .msg = DecodeTrace.msg },
        .teacher = if (a.teacher != null) .{ .ctx = &teacher, .input = GoldenTeacher.input, .mlp = GoldenTeacher.mlp, .msg = GoldenTeacher.msg } else null,
    };

    const metas = try gpa.alloc(ink2.check_rows.StepMeta, a.decode + 1);
    defer gpa.free(metas);
    var pos: usize = 0;
    for (steps[0 .. a.decode + 1], 0..) |ids, s| {
        tracer.step = s;
        tracer.step_pos0 = pos;
        teacher.step = s;
        teacher.step_pos0 = pos;
        metas[s] = .{ .n = ids.len, .pos0 = pos };
        var pick: u32 = undefined;
        if (s == 0) {
            try ink2.prefill.run(c, session.stream, 0, ids, a.chunk_rows, head_run.scratch.r.logits, hooks);
            pick = try head_run.finish();
        } else {
            try ink2.decode.step(c, session.stream, 0, ids[0], pos, sched.n - 1, hooks);
            pick = try head_run.run(0);
        }
        if (teacher.err) |e| return e;
        var name: [64]u8 = undefined;
        try head_run.put(&rows_out, try std.fmt.bufPrint(&name, "step{d}/logits", .{s}));
        try rows_out.put(try std.fmt.bufPrint(&name, "step{d}/argmax", .{s}), .u32, &.{1}, std.mem.asBytes(&pick));
        pos += ids.len;
    }

    try dumpKv(gpa, &rows_out, &session.cache, &sched, &session.w.cfg.text, kv, sched.n - 1, pos);
    var qn: [ink2.visits.max_visits]u32 = undefined;
    var loop_visits: [ink2.visits.max_visits]u32 = undefined;
    var n_loop: usize = 0;
    for (qn[0..sched.n], 0..) |*q, i| {
        q.* = ink2.decode.carriedRows(&sched, @intCast(i));
        if (sched.visits[i].pass > 0) {
            loop_visits[n_loop] = @intCast(i);
            n_loop += 1;
        }
    }
    try rows_out.finishRun(.{ .hidden = loop_hidden, .visits = sched.n, .unpadded_vocab = vocab, .qn = qn[0..sched.n], .loop_visits = loop_visits[0..n_loop], .steps = metas });
    const extra = sessionBytes(&session, ink2.head.deviceBytes(&session.w.cfg.text, 1).scratch);
    const caches = session.cache.kv.len + session.cache.conv.len;
    std.debug.print("prefill: {d} ids in chunks of {d}, {d} decode steps, device_bytes {d} + {d} (caches {d}, scratch {d}) = {d}\n", .{ steps[0].len, a.chunk_rows, a.decode, session.w.device_bytes, extra, caches, extra - caches, session.w.device_bytes + extra });
}
