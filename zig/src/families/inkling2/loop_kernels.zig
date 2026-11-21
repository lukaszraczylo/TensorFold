//! Inkling 2 Mini's loop kernels (inkling2/loop.cu): the image's module and its functions.
const std = @import("std");
const cuda = @import("cuda");

/// Threads a row's quantiser block runs; loop.cu's kThreads.
pub const actq_threads = 256;
pub const nvfp4_block = 16;
pub const mxfp8_block = 32;

pub const Kernels = struct {
    mod: cuda.Module,
    fill: cuda.Function,
    actq_nvfp4: cuda.Function,
    actq_mxfp8: cuda.Function,
    actq_nvfp4_codes: cuda.Function,
    actq_mxfp8_codes: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .fill = try mod.function("tf_ink2l_fill"),
            .actq_nvfp4 = try mod.function("tf_ink2l_actq_nvfp4"),
            .actq_mxfp8 = try mod.function("tf_ink2l_actq_mxfp8"),
            .actq_nvfp4_codes = try mod.function("tf_ink2l_actq_nvfp4_codes"),
            .actq_mxfp8_codes = try mod.function("tf_ink2l_actq_mxfp8_codes"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// `extra` are the codes kernels' output pointers, between `out` and the sizes.
fn actq(f: cuda.Function, block: usize, s: cuda.Stream, x: u64, out: u64, extra: []const u64, rows: usize, kdim: usize) !void {
    if (kdim == 0 or kdim % block != 0) return error.BadWidth;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    args.add(x);
    args.add(out);
    for (extra) |p| args.add(p);
    args.add(@as(i32, @intCast(rows)));
    args.add(@as(i32, @intCast(kdim)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = actq_threads } }, s, &args);
}

/// `x` is bf16 [rows, kdim], `out` float32 [rows, kdim]: per-row NVFP4 with the four-over-six rule, dequantised.
pub fn actNvfp4(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_nvfp4, nvfp4_block, s, x, out, &.{}, rows, kdim);
}

/// Same shapes as `actNvfp4`: per-32 MXFP8 with power-of-two scales.
pub fn actMxfp8(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_mxfp8, mxfp8_block, s, x, out, &.{}, rows, kdim);
}

/// As `actNvfp4`, plus ref.quant.nvfp4_codes: e2m1 values float32 [rows, kdim], e4m3 block scales float32 [rows, kdim/16], row amax float32 [rows].
pub fn actNvfp4Codes(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, values: u64, scales: u64, amax: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_nvfp4_codes, nvfp4_block, s, x, out, &.{ values, scales, amax }, rows, kdim);
}

/// As `actMxfp8`, plus ref.quant.mxfp8_codes: e4m3 values float32 [rows, kdim] and UE8M0 biased exponents u8 [rows, kdim/32].
pub fn actMxfp8Codes(k: *const Kernels, s: cuda.Stream, x: u64, out: u64, values: u64, exps: u64, rows: usize, kdim: usize) !void {
    return actq(k.actq_mxfp8_codes, mxfp8_block, s, x, out, &.{ values, exps }, rows, kdim);
}

const Fake = struct {
    var grid_x: u32 = 0;
    var block_x: u32 = 0;
    var which: usize = 0;
    var count: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        grid_x = gx;
        block_x = bx;
        which = @intFromPtr(f);
        count += 1;
        return 0;
    }
};

test "actq launches one block a row and refuses a width off the block" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const fun = struct {
        fn of(drv: *const cuda.Driver, i: usize) cuda.Function {
            return .{ .d = drv, .handle = @ptrFromInt(i) };
        }
    };
    const k: Kernels = .{ .mod = .{ .d = &d, .handle = null }, .fill = fun.of(&d, 1), .actq_nvfp4 = fun.of(&d, 2), .actq_mxfp8 = fun.of(&d, 3), .actq_nvfp4_codes = fun.of(&d, 4), .actq_mxfp8_codes = fun.of(&d, 5) };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try actNvfp4(&k, s, 0x1000, 0x2000, 64, 2304);
    try std.testing.expectEqual(@as(u32, 64), Fake.grid_x);
    try std.testing.expectEqual(@as(u32, actq_threads), Fake.block_x);
    try std.testing.expectEqual(@as(usize, 2), Fake.which);
    try actMxfp8(&k, s, 0x1000, 0x2000, 3, 1152);
    try std.testing.expectEqual(@as(u32, 3), Fake.grid_x);
    try std.testing.expectEqual(@as(usize, 3), Fake.which);
    const launched = Fake.count;
    try std.testing.expectError(error.BadWidth, actNvfp4(&k, s, 0, 0, 1, 2312));
    try std.testing.expectError(error.BadWidth, actMxfp8(&k, s, 0, 0, 1, 1136));
    try std.testing.expectError(error.BadWidth, actNvfp4(&k, s, 0, 0, 1, 0));
    try std.testing.expectEqual(launched, Fake.count);
    try actNvfp4Codes(&k, s, 0, 0, 0, 0, 0, 5, 2304);
    try std.testing.expectEqual(@as(usize, 4), Fake.which);
    try std.testing.expectEqual(@as(u32, 5), Fake.grid_x);
    try actMxfp8Codes(&k, s, 0, 0, 0, 0, 7, 1152);
    try std.testing.expectEqual(@as(usize, 5), Fake.which);
    try std.testing.expectError(error.BadWidth, actMxfp8Codes(&k, s, 0, 0, 0, 0, 1, 1136));
    // 1136 is a multiple of 16 but not of 32
    try actNvfp4(&k, s, 0, 0, 1, 1136);
}

test "the loader type-checks in a build with kernels" {
    _ = &Kernels.load;
    _ = &Kernels.unload;
}
