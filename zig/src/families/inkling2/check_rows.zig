//! Rows for a reference to compare: one raw file a tensor plus a manifest.json of {name, dtype, shape}.
const std = @import("std");

pub const Dtype = enum {
    f32,
    bf16,
    u8,
    u32,
    i32,

    pub fn size(d: Dtype) usize {
        return switch (d) {
            .f32, .u32, .i32 => 4,
            .bf16 => 2,
            .u8 => 1,
        };
    }
};

pub const manifest_name = "manifest.json";

pub const Writer = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    manifest: std.ArrayList(u8),
    count: usize,

    /// `dir` is created with its parents when missing.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Writer {
        const cwd = std.Io.Dir.cwd();
        try cwd.createDirPath(io, dir);
        return .{ .gpa = gpa, .io = io, .dir = try cwd.openDir(io, dir, .{}), .manifest = .empty, .count = 0 };
    }

    pub fn deinit(w: *Writer) void {
        w.manifest.deinit(w.gpa);
        w.dir.close(w.io);
    }

    /// Writes `<name>.<dtype>` and records it; `bytes` must hold exactly the shape's elements.
    pub fn put(w: *Writer, name: []const u8, dtype: Dtype, shape: []const usize, bytes: []const u8) !void {
        if (!validName(name)) return error.InvalidName;
        var elems: usize = 1;
        for (shape) |n| elems *= n;
        if (bytes.len != elems * dtype.size()) return error.ShapeMismatch;
        const file = try std.fmt.allocPrint(w.gpa, "{s}.{s}", .{ name, @tagName(dtype) });
        defer w.gpa.free(file);
        if (std.fs.path.dirname(file)) |parent| try w.dir.createDirPath(w.io, parent);
        try w.dir.writeFile(w.io, .{ .sub_path = file, .data = bytes });
        try w.manifest.appendSlice(w.gpa, if (w.count == 0) "[\n" else ",\n");
        try w.manifest.print(w.gpa, "{{\"name\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ name, @tagName(dtype) });
        for (shape, 0..) |n, i| try w.manifest.print(w.gpa, "{s}{d}", .{ if (i == 0) "" else ",", n });
        try w.manifest.appendSlice(w.gpa, "]}");
        w.count += 1;
    }

    pub fn finish(w: *Writer) !void {
        if (w.count == 0) try w.manifest.appendSlice(w.gpa, "[");
        try w.manifest.appendSlice(w.gpa, "\n]\n");
        try w.dir.writeFile(w.io, .{ .sub_path = manifest_name, .data = w.manifest.items });
    }
};

/// A relative path of plain characters: it stays under the directory and needs no JSON escaping.
fn validName(name: []const u8) bool {
    if (name.len == 0 or name[0] == '/') return false;
    var parts = std.mem.splitScalar(u8, name, '/');
    while (parts.next()) |p| {
        if (p.len == 0 or std.mem.eql(u8, p, ".") or std.mem.eql(u8, p, "..")) return false;
        for (p) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-' and c != '.') return false;
    }
    return true;
}

test "writer lays out files and manifest" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/rows", .{&tmp.sub_path});
    defer gpa.free(out);

    var w = try Writer.init(gpa, io, out);
    defer w.deinit();
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try w.put("a/b", .f32, &.{ 2, 3 }, std.mem.sliceAsBytes(&vals));
    try w.finish();

    const data = try tmp.dir.readFileAlloc(io, "rows/a/b.f32", gpa, .limited(1 << 10));
    defer gpa.free(data);
    try std.testing.expectEqual(@as(usize, 24), data.len);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&vals), data);

    const text = try tmp.dir.readFileAlloc(io, "rows/" ++ manifest_name, gpa, .limited(1 << 10));
    defer gpa.free(text);
    const Entry = struct { name: []const u8, dtype: []const u8, shape: []const usize };
    const parsed = try std.json.parseFromSlice([]const Entry, gpa, text, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 1), parsed.value.len);
    try std.testing.expectEqualStrings("a/b", parsed.value[0].name);
    try std.testing.expectEqualStrings("f32", parsed.value[0].dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, parsed.value[0].shape);
}

test "writer refuses a wrong byte count and an escaping name" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/rows", .{&tmp.sub_path});
    defer gpa.free(out);
    var w = try Writer.init(gpa, io, out);
    defer w.deinit();
    try std.testing.expectError(error.ShapeMismatch, w.put("x", .u32, &.{2}, &.{ 0, 0, 0, 0 }));
    try std.testing.expectError(error.InvalidName, w.put("../x", .u8, &.{1}, &.{0}));
    try std.testing.expectError(error.InvalidName, w.put("a\"b", .u8, &.{1}, &.{0}));
}
