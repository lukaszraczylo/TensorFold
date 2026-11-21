//! `tf-ink2-check <subcommand> --out DIR [options]`: rows of Inkling 2 Mini's CUDA engine for a reference to compare.
const std = @import("std");
const cuda = @import("cuda");
const ink2 = @import("inkling2");

const usage =
    \\usage: tf-ink2-check <subcommand> --out DIR [options]
    \\smoke: fills a device buffer with tf_ink2l_fill and writes DIR/smoke.u32 (1024 words of 0xC0FFEE)
    \\
;

pub const smoke_words = 1024;
pub const smoke_value: u32 = 0xC0FFEE;
const smoke_threads = 256;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or !std.mem.eql(u8, args[2], "--out")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    if (std.mem.eql(u8, args[1], "smoke")) {
        try smoke(init, args[3]);
        return 0;
    }
    std.debug.print("unknown subcommand: {s}\n{s}", .{ args[1], usage });
    return 2;
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
