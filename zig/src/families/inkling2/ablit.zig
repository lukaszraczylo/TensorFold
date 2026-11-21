//! TF_INK2_ABLIT: the residual edit file, its device state and the kernel launch after an attention output projection.
const std = @import("std");
const cuda = @import("cuda");
const platform = @import("sys");
const loop_kernels = @import("loop_kernels.zig");

pub const magic = "TFINK2A1";
pub const max_scale: f32 = 16.0;
pub const max_rank = 8;
const header_bytes = magic.len + 3 * 4;
const ortho_tol = 1e-4;
const path_env = "TF_INK2_ABLIT";
const scale_env = "TF_INK2_ABLIT_SCALE";
const off_text = "0";
const block_threads = loop_kernels.actq_threads;

pub const Error = error{ BadSize, BadMagic, BadShape, NotFinite, NegativeScale, NotOrthonormal, OutOfMemory };

pub const Edit = struct {
    hidden: u32,
    layers: u32,
    rank: u32,
    q: []f32, // [hidden, rank] row-major
    scales: []f32, // [layers]

    pub fn deinit(e: Edit, gpa: std.mem.Allocator) void {
        gpa.free(e.q);
        gpa.free(e.scales);
    }
};

fn u32At(bytes: []const u8, at: usize) u32 {
    return std.mem.readInt(u32, bytes[at..][0..4], .little);
}

fn f32At(bytes: []const u8, at: usize) f32 {
    return @bitCast(u32At(bytes, at));
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, hidden: u32, layers: u32) Error!Edit {
    if (bytes.len < header_bytes) return error.BadSize;
    if (!std.mem.eql(u8, bytes[0..magic.len], magic)) return error.BadMagic;
    const rank = u32At(bytes, magic.len + 8);
    if (u32At(bytes, magic.len) != hidden or u32At(bytes, magic.len + 4) != layers or rank < 1 or rank > max_rank) return error.BadShape;
    const nq = @as(usize, hidden) * rank;
    if (bytes.len != header_bytes + 4 * (nq + layers)) return error.BadSize;
    const q = try gpa.alloc(f32, nq);
    errdefer gpa.free(q);
    const scales = try gpa.alloc(f32, layers);
    errdefer gpa.free(scales);
    for (q, 0..) |*v, i| {
        v.* = f32At(bytes, header_bytes + 4 * i);
        if (!std.math.isFinite(v.*)) return error.NotFinite;
    }
    for (scales, 0..) |*v, i| {
        v.* = f32At(bytes, header_bytes + 4 * (nq + i));
        if (!std.math.isFinite(v.*)) return error.NotFinite;
        if (v.* < 0) return error.NegativeScale;
    }
    for (0..rank) |a| for (a..rank) |b| {
        var dot: f64 = 0;
        for (0..hidden) |i| dot += @as(f64, q[i * rank + a]) * q[i * rank + b];
        if (@abs(dot - @as(f64, @floatFromInt(@intFromBool(a == b)))) > ortho_tol) return error.NotOrthonormal;
    };
    return .{ .hidden = hidden, .layers = layers, .rank = rank, .q = q, .scales = scales };
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, hidden: u32, layers: u32) !Edit {
    const max = header_bytes + 4 * (@as(usize, hidden) * max_rank + layers);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(max + 1));
    defer gpa.free(bytes);
    return parse(gpa, bytes, hidden, layers);
}

/// The edit file's path from the setting's raw text: unset, blank or "0" is off.
pub fn pathOf(raw: ?[]const u8) platform.environment.Error!?[]const u8 {
    const p = (try platform.environment.path(path_env, raw)) orelse return null;
    return if (std.mem.eql(u8, p, off_text)) null else p;
}

pub const State = struct {
    q: cuda.DeviceBuffer,
    table: cuda.DeviceBuffer, // table[l] = scales[l] * multiplier
    host: []f32, // the same table: a zero entry launches nothing
    rank: u32,
};

/// The edit on the device, its scales times `mult`.
pub fn upload(gpa: std.mem.Allocator, d: *const cuda.Driver, file: Edit, mult: f32) !State {
    const host = try gpa.alloc(f32, file.scales.len);
    errdefer gpa.free(host);
    for (host, file.scales) |*h, s| h.* = s * mult;
    var q = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(file.q));
    errdefer q.free();
    var table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
    errdefer table.free();
    return .{ .q = q, .table = table, .host = host, .rank = file.rank };
}

/// Loads the edit when TF_INK2_ABLIT names a file; null leaves the engine untouched.
pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, hidden: u32, layers: u32) !?State {
    const path = (try pathOf(platform.environment.Process.get(.{}, path_env))) orelse return null;
    const mult = try platform.environment.read(f32, scale_env, 1.0);
    const file = try load(gpa, io, path, hidden, layers);
    defer file.deinit(gpa);
    const st = try upload(gpa, d, file, mult);
    std.debug.print("info: residual edit {s}, rank {d}, scale x{d}\n", .{ path, file.rank, mult });
    return st;
}

pub fn close(st: *State, gpa: std.mem.Allocator) void {
    st.q.free();
    st.table.free();
    gpa.free(st.host);
    st.* = undefined;
}

/// `rows` rows of bf16 hidden values at `y`, an attention output projection of `layer`.
pub fn apply(k: *const loop_kernels.Kernels, st: *const State, s: cuda.Stream, layer: u32, y: u64, rows: usize, hidden: u32) !void {
    if (layer >= st.host.len) return error.BadLayer;
    if (rows == 0 or st.host[layer] == 0) return;
    var args: cuda.Args = .{};
    args.add(st.q.ptr);
    args.add(st.table.ptr);
    args.add(y);
    args.add(@as(i32, @intCast(hidden)));
    args.add(@as(i32, @intCast(st.rank)));
    args.add(@as(i32, @intCast(layer)));
    try cuda.launch.launch(k.ablit, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = block_threads } }, s, &args);
}

const t = std.testing;
const mini_hidden = 2304;
const mini_layers = 24;
const mini_scales = [mini_layers]f32{ 0, 0, 0, 0, 4, 4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4, 4, 4, 4, 4, 4 };

/// The Mini file image: Q's first two basis vectors, `col_scale` applied to column 1.
fn image(gpa: std.mem.Allocator, rank: u32, col_scale: f32) ![]u8 {
    const nq = mini_hidden * rank;
    const out = try gpa.alloc(u8, header_bytes + 4 * (nq + mini_layers));
    @memset(out, 0);
    @memcpy(out[0..magic.len], magic);
    std.mem.writeInt(u32, out[8..12], mini_hidden, .little);
    std.mem.writeInt(u32, out[12..16], mini_layers, .little);
    std.mem.writeInt(u32, out[16..20], rank, .little);
    for (0..rank) |j| {
        const v: f32 = if (j == 1) col_scale else 1.0;
        std.mem.writeInt(u32, out[header_bytes + 4 * (j * rank + j) ..][0..4], @bitCast(v), .little);
    }
    for (mini_scales, 0..) |s, l| std.mem.writeInt(u32, out[header_bytes + 4 * (nq + l) ..][0..4], @bitCast(s), .little);
    return out;
}

fn setF32(bytes: []u8, at: usize, v: f32) void {
    std.mem.writeInt(u32, bytes[at..][0..4], @bitCast(v), .little);
}

test "parse accepts the Mini file" {
    const img = try image(t.allocator, 2, 1.0);
    defer t.allocator.free(img);
    try t.expectEqual(@as(usize, 18548), img.len);
    const e = try parse(t.allocator, img, mini_hidden, mini_layers);
    defer e.deinit(t.allocator);
    try t.expectEqual(@as(u32, 2), e.rank);
    try t.expectEqualSlices(f32, &mini_scales, e.scales);
    try t.expectEqual(@as(f32, 1.0), e.q[0]);
    try t.expectEqual(@as(f32, 1.0), e.q[3]);
}

test "parse refuses each bad file with its own error" {
    const img = try image(t.allocator, 2, 1.0);
    defer t.allocator.free(img);
    const p = struct {
        fn run(bytes: []const u8, hidden: u32) Error!Edit {
            return parse(t.allocator, bytes, hidden, mini_layers);
        }
    };
    try t.expectError(error.BadSize, p.run(img[0 .. img.len - 1], mini_hidden));
    try t.expectError(error.BadSize, p.run(img[0 .. header_bytes - 1], mini_hidden));
    try t.expectError(error.BadShape, p.run(img, mini_hidden + 1));
    const bad = try t.allocator.dupe(u8, img);
    defer t.allocator.free(bad);
    @memcpy(bad[0..magic.len], "TFINK2B1");
    try t.expectError(error.BadMagic, p.run(bad, mini_hidden));
    @memcpy(bad[0..magic.len], magic);
    std.mem.writeInt(u32, bad[16..20], 0, .little);
    try t.expectError(error.BadShape, p.run(bad, mini_hidden));
    std.mem.writeInt(u32, bad[16..20], max_rank + 1, .little);
    try t.expectError(error.BadShape, p.run(bad, mini_hidden));
    @memcpy(bad, img);
    setF32(bad, header_bytes, std.math.nan(f32));
    try t.expectError(error.NotFinite, p.run(bad, mini_hidden));
    @memcpy(bad, img);
    setF32(bad, img.len - 4, -0.5);
    try t.expectError(error.NegativeScale, p.run(bad, mini_hidden));
    setF32(bad, img.len - 4, std.math.inf(f32));
    try t.expectError(error.NotFinite, p.run(bad, mini_hidden));
    const scaled = try image(t.allocator, 2, 1.001);
    defer t.allocator.free(scaled);
    try t.expectError(error.NotOrthonormal, p.run(scaled, mini_hidden));
}

test "parse refuses a non-orthogonal pair of unit columns" {
    const img = try image(t.allocator, 2, 1.0);
    defer t.allocator.free(img);
    setF32(img, header_bytes + 4 * (0 * 2 + 1), 0.01);
    try t.expectError(error.NotOrthonormal, parse(t.allocator, img, mini_hidden, mini_layers));
}

test "parse takes rank 1 and rank 8" {
    for ([_]u32{ 1, 8 }) |r| {
        const img = try image(t.allocator, r, 1.0);
        defer t.allocator.free(img);
        const e = try parse(t.allocator, img, mini_hidden, mini_layers);
        defer e.deinit(t.allocator);
        try t.expectEqual(r, e.rank);
    }
}

test "env off" {
    const env = platform.environment;
    try t.expectEqual(@as(?[]const u8, null), try pathOf(null));
    try t.expectEqual(@as(?[]const u8, null), try pathOf(""));
    try t.expectEqual(@as(?[]const u8, null), try env.path(path_env, ""));
    try t.expectEqual(@as(?[]const u8, null), try pathOf("0"));
    try t.expectEqualStrings("/a/b", (try pathOf("/a/b")).?);
    try t.expectError(error.InvalidEnvironment, pathOf("/a\nb"));
}

test "scale env" {
    const env = platform.environment;
    try t.expectEqual(@as(f32, 1.0), try env.value(f32, scale_env, null, 1.0));
    try t.expectEqual(@as(f32, 2.5), try env.value(f32, scale_env, "2.5", 1.0));
    try t.expectEqual(@as(f32, max_scale), try env.value(f32, scale_env, "16", 1.0));
    for ([_][]const u8{ "0", "-1", "16.5", "nan", "inf", "abc" }) |bad|
        try t.expectError(error.InvalidEnvironment, env.value(f32, scale_env, bad, 1.0));
}

test "embed host env" {
    const env = platform.environment;
    try t.expectEqual(false, try env.value(bool, "TF_INK2_EMBED_HOST", null, false));
    try t.expectEqual(true, try env.value(bool, "TF_INK2_EMBED_HOST", "1", false));
    try t.expectError(error.InvalidEnvironment, env.value(bool, "TF_INK2_EMBED_HOST", "x", false));
}

test "rows registered" {
    for ([_][:0]const u8{ path_env, scale_env, "TF_INK2_EMBED_HOST" }) |name|
        try t.expectEqual(platform.environment.Scope.cuda, platform.environment.rule(name).scope);
}

test "apply launches a block a row and nothing for a zero scale" {
    const Fake = struct {
        var grid_x: u32 = 0;
        var count: usize = 0;
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
            grid_x = gx;
            count += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    var k: loop_kernels.Kernels = undefined;
    k.ablit = .{ .d = &d, .handle = @ptrFromInt(1) };
    var host = [_]f32{ 0, 4 };
    const st: State = .{ .q = undefined, .table = undefined, .host = &host, .rank = 2 };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try apply(&k, &st, s, 0, 0x1000, 4, mini_hidden);
    try t.expectEqual(@as(usize, 0), Fake.count);
    try apply(&k, &st, s, 1, 0x1000, 4, mini_hidden);
    try t.expectEqual(@as(usize, 1), Fake.count);
    try t.expectEqual(@as(u32, 4), Fake.grid_x);
    try t.expectError(error.BadLayer, apply(&k, &st, s, 2, 0x1000, 4, mini_hidden));
}
