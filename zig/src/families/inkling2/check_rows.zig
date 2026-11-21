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

/// What a run's manifest.json carries beside its files, in `ref.capture.write_golden`'s layout.
pub const RunMeta = struct {
    hidden: usize,
    visits: usize,
    unpadded_vocab: usize,
    qn: []const u32,
    steps: []const StepMeta,
    /// Visits with pass > 0, the ones that carry messages and controller outputs.
    loop_visits: []const u32 = &.{},
};

pub const StepMeta = struct { n: usize, pos0: usize };

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
        try w.closeFiles();
        try w.dir.writeFile(w.io, .{ .sub_path = manifest_name, .data = w.manifest.items });
    }

    /// As `finish`, but manifest.json is the run's dict: the reference's fields and the written rows under `files`.
    pub fn finishRun(w: *Writer, meta: RunMeta) !void {
        try w.closeFiles();
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(w.gpa);
        try out.print(w.gpa, "{{\"hidden\":{d},\"visits\":{d},\"unpadded_vocab\":{d},\"qn\":[", .{ meta.hidden, meta.visits, meta.unpadded_vocab });
        for (meta.qn, 0..) |q, i| try out.print(w.gpa, "{s}{d}", .{ if (i == 0) "" else ",", q });
        try out.appendSlice(w.gpa, "],\"loop_visits\":[");
        for (meta.loop_visits, 0..) |v, i| try out.print(w.gpa, "{s}{d}", .{ if (i == 0) "" else ",", v });
        try out.appendSlice(w.gpa, "],\"steps\":[");
        for (meta.steps, 0..) |st, i| try out.print(w.gpa, "{s}{{\"n\":{d},\"pos0\":{d}}}", .{ if (i == 0) "" else ",", st.n, st.pos0 });
        try out.appendSlice(w.gpa, "],\"files\":");
        try out.appendSlice(w.gpa, w.manifest.items);
        try out.appendSlice(w.gpa, "}\n");
        try w.dir.writeFile(w.io, .{ .sub_path = manifest_name, .data = out.items });
    }

    fn closeFiles(w: *Writer) !void {
        if (w.count == 0) try w.manifest.appendSlice(w.gpa, "[");
        try w.manifest.appendSlice(w.gpa, "\n]\n");
    }
};

/// ref.capture's all-rows layout [rows, qn, width] from the engine's [qn, rows, width].
pub fn toFileLayout(dev: []const f32, out: []f32, qn: usize, rows: usize, width: usize) void {
    for (0..rows) |r| for (0..qn) |q| @memcpy(out[(r * qn + q) * width ..][0..width], dev[(q * rows + r) * width ..][0..width]);
}

/// The engine's [qn, rows, width] from the file layout [rows, qn, width].
pub fn toDeviceLayout(file: []const f32, out: []f32, qn: usize, rows: usize, width: usize) void {
    for (0..rows) |r| for (0..qn) |q| @memcpy(out[(q * rows + r) * width ..][0..width], file[(r * qn + q) * width ..][0..width]);
}

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

test "finishRun writes the reference's manifest layout with the rows under files" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/run", .{&tmp.sub_path});
    defer gpa.free(out);

    var w = try Writer.init(gpa, io, out);
    defer w.deinit();
    const vals = [_]f32{ 1, 2 };
    try w.put("step0/visit00", .f32, &.{ 1, 2 }, std.mem.sliceAsBytes(&vals));
    try w.finishRun(.{ .hidden = 2, .visits = 3, .unpadded_vocab = 9, .qn = &.{ 1, 3, 1 }, .loop_visits = &.{2}, .steps = &.{ .{ .n = 1, .pos0 = 0 }, .{ .n = 1, .pos0 = 1 } } });

    const data = try tmp.dir.readFileAlloc(io, "run/step0/visit00.f32", gpa, .limited(1 << 10));
    defer gpa.free(data);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&vals), data);
    const text = try tmp.dir.readFileAlloc(io, "run/" ++ manifest_name, gpa, .limited(1 << 10));
    defer gpa.free(text);
    const Entry = struct { name: []const u8, dtype: []const u8, shape: []const usize };
    const Manifest = struct { hidden: usize, visits: usize, unpadded_vocab: usize, qn: []const u32, loop_visits: []const u32, steps: []const StepMeta, files: []const Entry };
    const parsed = try std.json.parseFromSlice(Manifest, gpa, text, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.value.hidden);
    try std.testing.expectEqual(@as(usize, 3), parsed.value.visits);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 1 }, parsed.value.qn);
    try std.testing.expectEqualSlices(u32, &.{2}, parsed.value.loop_visits);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.steps[1].pos0);
    try std.testing.expectEqualStrings("step0/visit00", parsed.value.files[0].name);
}

test "finishRun with no rows keeps files an empty array" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/run", .{&tmp.sub_path});
    defer gpa.free(out);
    var w = try Writer.init(gpa, io, out);
    defer w.deinit();
    try w.finishRun(.{ .hidden = 2, .visits = 0, .unpadded_vocab = 0, .qn = &.{}, .steps = &.{} });
    const text = try tmp.dir.readFileAlloc(io, "run/" ++ manifest_name, gpa, .limited(1 << 10));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(struct { files: []const u8 }, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed.value.files.len);
}

test "row layouts swap the row and query axes and invert each other" {
    const qn = 3;
    const rows = 4;
    const width = 2;
    var dev: [qn * rows * width]f32 = undefined;
    for (&dev, 0..) |*x, i| x.* = @floatFromInt(i);
    var file: [dev.len]f32 = undefined;
    toFileLayout(&dev, &file, qn, rows, width);
    // device (q = 1, r = 2) sits at (2 * qn + 1) in the file
    try std.testing.expectEqualSlices(f32, dev[(1 * rows + 2) * width ..][0..width], file[(2 * qn + 1) * width ..][0..width]);
    var back: [dev.len]f32 = undefined;
    toDeviceLayout(&file, &back, qn, rows, width);
    try std.testing.expectEqualSlices(f32, &dev, &back);
    // one query: both layouts are the same bytes
    toFileLayout(dev[0 .. rows * width], file[0 .. rows * width], 1, rows, width);
    try std.testing.expectEqualSlices(f32, dev[0 .. rows * width], file[0 .. rows * width]);
}
