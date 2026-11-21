//! `tf-ink2-check <subcommand> --out DIR [options]`: rows of Inkling 2 Mini's CUDA engine for a reference to compare.
const std = @import("std");
const cuda = @import("cuda");
const ink2 = @import("inkling2");

const usage =
    \\usage: tf-ink2-check <subcommand> --out DIR [options]
    \\smoke: fills a device buffer with tf_ink2l_fill and writes DIR/smoke.u32 (1024 words of 0xC0FFEE)
    \\weights --model DIR [--layers A-B] [--readback NAME,...]: uploads the text weights (TF_INK2_EMBED_HOST=1 keeps the
    \\    embedding on the host), writes DIR/weights.json with peak_rss_mib and device_bytes, and one row per readback
    \\    tensor (names below model.llm., e.g. layers.0.attn.wo_ud.weight)
    \\
;

const embed_host_env = "TF_INK2_EMBED_HOST";
const dev_context = 2048;
const dev_chunk_rows = 64;
const device_limit: u64 = 30 << 30; // ASSUMPTION: the dev pod's GB10 pool shows about 31 GiB available
const stats_name = "weights.json";

const Args = struct {
    sub: []const u8,
    out: ?[]const u8 = null,
    model: ?[]const u8 = null,
    layers: ?ink2.budget.LayerRange = null,
    readback: ?[]const u8 = null,
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
        if (std.mem.eql(u8, args[i], "--out")) a.out = v else if (std.mem.eql(u8, args[i], "--model")) a.model = v else if (std.mem.eql(u8, args[i], "--layers")) a.layers = try parseLayers(v) else if (std.mem.eql(u8, args[i], "--readback")) a.readback = v else return error.Usage;
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

    const embed_host = if (init.environ_map.get(embed_host_env)) |v| std.mem.eql(u8, v, "1") else false;
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
