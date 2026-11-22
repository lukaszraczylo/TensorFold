//! Host mirror of the 8-bit attention cache's row format (ink2_kv8.cuh): a head's 128 bf16 values as int8 codes and one fp32 scale.
const std = @import("std");
const kernels = @import("kernels.zig");
const decode = @import("decode.zig");

const dim = kernels.head_dim;
const code_max: f32 = kernels.kv8_code_max;

/// Round to nearest, ties to even (`__float2int_rn`).
fn rintEven(x: f32) f32 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5) 2.0 * @round(x / 2.0) else r;
}

/// The codes of `x` (bf16 values widened) and the row's scale, as tf_ink2_prep_q8 writes them; an all-zero row has scale 0.
pub fn quantRow(codes: *[dim]i8, x: *const [dim]f32) f32 {
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(v));
    const inv: f32 = if (amax > 0) code_max / amax else 0;
    for (codes, x) |*c, v| c.* = @intFromFloat(std.math.clamp(rintEven(v * inv), -code_max, code_max));
    return amax / code_max;
}

/// The bf16 bits an attention kernel's tile holds for a row's codes: code * scale, rounded once to bf16.
pub fn dequantRow(out: *[dim]u16, codes: *const [dim]i8, scale: f32) void {
    for (out, codes) |*o, c| o.* = decode.bf16Bits(@as(f32, @floatFromInt(c)) * scale);
}

fn widen(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

const t = std.testing;

test "round trip stays within half a code step" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (0..64) |_| {
        var x: [dim]f32 = undefined;
        for (&x) |*v| v.* = widen(decode.bf16Bits(rnd.floatNorm(f32) * 3.0));
        var codes: [dim]i8 = undefined;
        const scale = quantRow(&codes, &x);
        var amax: f32 = 0;
        for (x) |v| amax = @max(amax, @abs(v));
        try t.expectEqual(amax / 127.0, scale);
        var back: [dim]u16 = undefined;
        dequantRow(&back, &codes, scale);
        for (x, back, codes) |v, b, c| {
            try t.expect(@abs(c) <= 127);
            // half a step, plus the bf16 rounding of the product (2^-8 relative)
            try t.expect(@abs(widen(b) - v) <= scale * 0.5 + @abs(v) * (1.0 / 256.0));
        }
    }
}

test "the largest value maps to a full-scale code and keeps its sign" {
    var x: [dim]f32 = @splat(0.0);
    x[5] = -2.0;
    x[9] = 1.0;
    var codes: [dim]i8 = undefined;
    const scale = quantRow(&codes, &x);
    try t.expectEqual(@as(i8, -127), codes[5]);
    try t.expectEqual(@as(i8, 64), codes[9]); // 63.5 ties to the even 64
    try t.expectEqual(@as(i8, 0), codes[0]);
    var back: [dim]u16 = undefined;
    dequantRow(&back, &codes, scale);
    try t.expectEqual(@as(f32, -2.0), widen(back[5]));
}

test "a zero row has scale zero and zero codes" {
    const x: [dim]f32 = @splat(0.0);
    var codes: [dim]i8 = undefined;
    try t.expectEqual(@as(f32, 0.0), quantRow(&codes, &x));
    for (codes) |c| try t.expectEqual(@as(i8, 0), c);
    var back: [dim]u16 = undefined;
    dequantRow(&back, &codes, 0.0);
    for (back) |b| try t.expectEqual(@as(u16, 0), b);
}

test "an outlier costs the other values resolution, not the outlier" {
    var x: [dim]f32 = @splat(0.01);
    x[0] = 100.0;
    var codes: [dim]i8 = undefined;
    const scale = quantRow(&codes, &x);
    try t.expectEqual(@as(i8, 127), codes[0]);
    try t.expectEqual(@as(i8, 0), codes[1]); // 0.01 / (100 / 127) rounds to 0
    try t.expect(scale > 0.78 and scale < 0.79);
}

test "ties round to even" {
    try t.expectEqual(@as(f32, 2.0), rintEven(2.5));
    try t.expectEqual(@as(f32, 4.0), rintEven(3.5));
    try t.expectEqual(@as(f32, -2.0), rintEven(-2.5));
    try t.expectEqual(@as(f32, 3.0), rintEven(2.6));
}
