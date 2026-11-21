//! The process's peak resident set from /proc/self/status.
const std = @import("std");

const path = "/proc/self/status";
const key = "VmHWM:";
const kib_per_mib = 1024;
const max_bytes = 64 << 10;

/// MiB of the `VmHWM:` line in `text`; the line's value is in kB.
pub fn parsePeakMib(text: []const u8) error{NoPeakRss}!u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        var f = std.mem.tokenizeAny(u8, line[key.len..], " \tkB");
        const kb = std.fmt.parseInt(u64, f.next() orelse return error.NoPeakRss, 10) catch return error.NoPeakRss;
        return kb / kib_per_mib;
    }
    return error.NoPeakRss;
}

/// procfs reports size 0, so the file is read to EOF rather than by its stat size.
pub fn peakMib(gpa: std.mem.Allocator, io: std.Io) !u64 {
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    const text = try r.interface.allocRemaining(gpa, .limited(max_bytes));
    defer gpa.free(text);
    return parsePeakMib(text) catch |e| {
        std.debug.print("{s}: no {s} line\n", .{ path, key });
        return e;
    };
}

test "parses the status line as procfs writes it" {
    const text = "Name:\ttf-ink2-check\nVmPeak:\t  999 kB\nVmHWM:\t    1824 kB\nVmRSS:\t 100 kB\n";
    try std.testing.expectEqual(@as(u64, 1), try parsePeakMib(text));
    try std.testing.expectEqual(@as(u64, 1536), try parsePeakMib("VmHWM:\t 1572864 kB\n"));
}

test "a missing or empty key is refused" {
    try std.testing.expectError(error.NoPeakRss, parsePeakMib(""));
    try std.testing.expectError(error.NoPeakRss, parsePeakMib("VmRSS:\t 5 kB\n"));
    try std.testing.expectError(error.NoPeakRss, parsePeakMib("VmHWM:\n"));
}
