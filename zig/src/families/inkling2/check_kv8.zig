//! `tf-ink2-check kv8`: the 8-bit attention cache's kernels against the bf16 ones on random rows, with no model.
//! Prep writes the same rows into a bf16 cache (A) and an 8-bit one (B). B's codes must equal the host quantiser's on A's rows.
//! A bf16 cache C holding B's widened rows then runs the bf16 kernels: every 8-bit attention path must match C bit for bit,
//! and stay near A within the quantisation error.
const std = @import("std");
const cuda = @import("cuda");
const ink2 = @import("inkling2");

const k = ink2.kernels;
const Layer = k.Layer;

const heads = 24;
const kv_heads = 8;
const dim = k.head_dim;
const hidden = 2304;
const context = 17408;
const filled = 17000;
const max_rows = 512;
const group = heads / kv_heads;
/// Largest L2 error of an attention output head vector, relative to the larger of its bf16-cache norm and its key head's mean V row norm
/// (random keys give near-uniform attention, whose output is far smaller than any V row, so its own norm alone would mislead).
const max_rel_err = 0.05;

const prep_consts: k.Prep = .{ .qk_clip = 22.6, .rel_clip = 8, .floor = 4096, .alpha = 0.1, .eps = 1e-6 };

const Case = struct {
    name: []const u8,
    p0: usize,
    rows: usize,
    streams: u32,
    seed: u64,
};

/// Disjoint position ranges: a decode row, a window, a mid-chunk row, a two-stream fold call and a prompt call over a chunk edge.
const cases = [_]Case{
    .{ .name = "decode at the end", .p0 = filled - 1, .rows = 1, .streams = 1, .seed = 11 },
    .{ .name = "window of 5", .p0 = filled - 10, .rows = 5, .streams = 1, .seed = 12 },
    .{ .name = "decode in chunk 0", .p0 = 9000, .rows = 1, .streams = 1, .seed = 13 },
    .{ .name = "fold, 128 rows", .p0 = 15000, .rows = 128, .streams = 2, .seed = 14 },
    .{ .name = "prompt, 512 rows over a chunk edge", .p0 = 16384 - 100, .rows = 512, .streams = 1, .seed = 15 },
};

fn widen(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// bf16 random rows: per-head magnitudes spread over a decade, one outlier in some rows.
const Raw = struct {
    q: []u16,
    k: []u16,
    v: []u16,
    rel: []u16,

    fn init(gpa: std.mem.Allocator, rows: usize) !Raw {
        const q = try gpa.alloc(u16, rows * heads * dim);
        errdefer gpa.free(q);
        const kk = try gpa.alloc(u16, rows * kv_heads * dim);
        errdefer gpa.free(kk);
        const v = try gpa.alloc(u16, rows * kv_heads * dim);
        errdefer gpa.free(v);
        const rel = try gpa.alloc(u16, rows * heads * k.d_rel);
        return .{ .q = q, .k = kk, .v = v, .rel = rel };
    }

    fn deinit(r: Raw, gpa: std.mem.Allocator) void {
        gpa.free(r.q);
        gpa.free(r.k);
        gpa.free(r.v);
        gpa.free(r.rel);
    }

    fn fill(r: Raw, rnd: std.Random, rows: usize, zero_head: bool) void {
        for (r.q) |*x| x.* = ink2.decode.bf16Bits(rnd.floatNorm(f32));
        for (r.rel) |*x| x.* = ink2.decode.bf16Bits(rnd.floatNorm(f32));
        for (0..rows) |row| for (0..kv_heads) |h| {
            const mag: f32 = 0.05 * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(h % 7)));
            const at = (row * kv_heads + h) * dim;
            for (r.k[at..][0..dim]) |*x| x.* = ink2.decode.bf16Bits(rnd.floatNorm(f32) * mag);
            for (r.v[at..][0..dim]) |*x| x.* = ink2.decode.bf16Bits(rnd.floatNorm(f32) * mag * 2);
            if (row % 7 == 0) r.k[at + 3] = ink2.decode.bf16Bits(widen(r.k[at + 3]) * 15);
            if (row % 5 == 0) r.v[at + 100] = ink2.decode.bf16Bits(widen(r.v[at + 100]) * 20);
            if (zero_head and row == 3 and h == 2) {
                @memset(r.k[at..][0..dim], 0);
                @memset(r.v[at..][0..dim], 0);
            }
        };
    }
};

const Rig = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    kern: k.Kernels,
    scratch: ink2.engine.Scratch,
    q_clip: cuda.DeviceBuffer,
    rel_clip: cuda.DeviceBuffer,
    proj: cuda.DeviceBuffer,

    fn put(rig: *const Rig, addr: u64, items: anytype) !void {
        const bytes = std.mem.sliceAsBytes(items);
        const view: cuda.DeviceBuffer = .{ .d = rig.d, .ptr = addr, .len = bytes.len, .borrowed = true };
        try view.uploadAsync(0, bytes, rig.s.handle);
    }

    /// Uploads a call's rows and runs prep into `layer`'s cache, then the bias table; q and rel stay in scratch.
    fn prep(rig: *const Rig, raw: Raw, p0: usize, rows: usize, streams: u32, layer: Layer, cache: k.Cache) !k.Rows {
        const r = rig.scratch.r;
        var pos: [max_rows]i32 = undefined;
        for (pos[0..rows], 0..) |*p, i| p.* = @intCast(p0 + i);
        var segs = [1]k.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p0), .off = 0 }};
        var tiles: [k.tileCount(max_rows, group) + 1]k.Tile = undefined;
        const n_tiles = try k.tiles(&segs, group, &tiles);
        try rig.put(r.q_raw, raw.q[0 .. rows * heads * dim]);
        try rig.put(r.k_conv, raw.k[0 .. rows * kv_heads * dim]);
        try rig.put(r.v_conv, raw.v[0 .. rows * kv_heads * dim]);
        try rig.put(r.r_raw, raw.rel[0 .. rows * heads * k.d_rel]);
        try rig.put(r.pos, pos[0..rows]);
        try rig.put(r.segs, &segs);
        try rig.put(r.tiles, tiles[0..n_tiles]);
        const call: k.Rows = .{
            .rows = @intCast(rows),
            .heads = heads,
            .kv_heads = kv_heads,
            .pos = r.pos,
            .seg_of = r.seg_of,
            .segs = r.segs,
            .tiles = r.tiles,
            .n_tiles = @intCast(n_tiles),
            .last_pos = @intCast(p0 + rows - 1),
            .streams = streams,
        };
        try k.prep(&rig.kern, rig.s, call, layer, prep_consts, .{ r.q_raw, r.k_conv, r.v_conv, r.r_raw }, .{ rig.q_clip.ptr, rig.rel_clip.ptr }, r.q, r.rel, cache);
        try k.relTable(&rig.kern, rig.s, call, r.rel, rig.proj.ptr, r.table);
        try rig.s.synchronize();
        return call;
    }

    /// The call's attention over `cache`, as bf16 [rows, heads * dim] bits.
    fn attend(rig: *const Rig, call: k.Rows, layer: Layer, cache: k.Cache, out: []u16) !void {
        const r = rig.scratch.r;
        const g = rig.scratch.geometry;
        try k.attend(&rig.kern, rig.s, call, layer, g.stride, g.slots(), r.q, cache, r.table, r.part_o, r.part_ml, r.sink, r.gate, r.out);
        try rig.s.synchronize();
        try cuda.memory.download(rig.d, r.out, std.mem.sliceAsBytes(out));
    }
};

fn randomBf16(gpa: std.mem.Allocator, rnd: std.Random, n: usize, lo: f32, hi: f32) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*x| x.* = ink2.decode.bf16Bits(lo + (hi - lo) * rnd.float(f32));
    return out;
}

fn cachePair(layer: Layer, base: u64) k.Cache {
    return .{ .k = base, .v = base + layer.cacheBytes(kv_heads) };
}

/// One cache's rows as bf16 [kv_heads, ring, dim] on the host: widened from codes when `layer` is 8-bit.
fn downloadBf16(gpa: std.mem.Allocator, d: *const cuda.Driver, at: u64, layer: Layer) ![]u16 {
    const rows = @as(usize, layer.ring) * kv_heads;
    const out = try gpa.alloc(u16, rows * dim);
    errdefer gpa.free(out);
    if (!layer.q8) {
        try cuda.memory.download(d, at, std.mem.sliceAsBytes(out));
        return out;
    }
    const raw = try gpa.alloc(u8, layer.cacheBytes(kv_heads));
    defer gpa.free(raw);
    try cuda.memory.download(d, at, raw);
    const scales = layer.scalesAt(kv_heads);
    for (0..rows) |slot| {
        const codes: *const [dim]i8 = @ptrCast(raw[slot * dim ..][0..dim]);
        const scale: f32 = @bitCast(std.mem.readInt(u32, raw[scales + slot * 4 ..][0..4], .little));
        ink2.kv8.dequantRow(out[slot * dim ..][0..dim], codes, scale);
    }
    return out;
}

/// Every (head, slot) row of B's K or V, requantised on the host from A's bf16 row, must equal B's codes and scale.
fn checkQuantised(gpa: std.mem.Allocator, d: *const cuda.Driver, a: []const u16, b_at: u64, lb: Layer, name: []const u8) !usize {
    const raw = try gpa.alloc(u8, lb.cacheBytes(kv_heads));
    defer gpa.free(raw);
    try cuda.memory.download(d, b_at, raw);
    const scales = lb.scalesAt(kv_heads);
    var bad: usize = 0;
    var zero_scales: usize = 0;
    for (0..@as(usize, lb.ring) * kv_heads) |slot| {
        var x: [dim]f32 = undefined;
        for (&x, a[slot * dim ..][0..dim]) |*o, u| o.* = widen(u);
        var codes: [dim]i8 = undefined;
        const scale = ink2.kv8.quantRow(&codes, &x);
        const got_codes: *const [dim]i8 = @ptrCast(raw[slot * dim ..][0..dim]);
        const got_scale: f32 = @bitCast(std.mem.readInt(u32, raw[scales + slot * 4 ..][0..4], .little));
        if (!std.mem.eql(i8, &codes, got_codes) or @as(u32, @bitCast(scale)) != @as(u32, @bitCast(got_scale))) bad += 1;
        if (scale == 0) zero_scales += 1;
    }
    std.debug.print("quantised {s}: {d} rows, {d} differ from the host quantiser, {d} zero-scale rows\n", .{ name, @as(usize, lb.ring) * kv_heads, bad, zero_scales });
    return bad;
}

pub fn run(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    var kern = try k.Kernels.load(ctx.d);
    defer kern.unload();

    const la = Layer.global(context);
    const lb = Layer.globalOf(context, true);
    const g: ink2.engine.Geometry = .{ .hidden = hidden, .heads = heads, .kv_heads = kv_heads, .dim = dim, .stride = k.slotsFor(la, context), .max_rows = max_rows };
    var scratch = try ink2.engine.Scratch.init(ctx.d, g);
    defer scratch.deinit();

    var prng = std.Random.DefaultPrng.init(2026);
    const rnd = prng.random();
    const q_clip_h = try randomBf16(gpa, rnd, heads * dim, 0.5, 1.5);
    defer gpa.free(q_clip_h);
    const rel_clip_h = try randomBf16(gpa, rnd, heads * k.d_rel, 0.5, 1.5);
    defer gpa.free(rel_clip_h);
    const proj_h = try randomBf16(gpa, rnd, k.d_rel * k.extent, -0.5, 0.5);
    defer gpa.free(proj_h);
    const sink_h = try randomBf16(gpa, rnd, max_rows * heads, -1, 1);
    defer gpa.free(sink_h);
    const gate_h = try randomBf16(gpa, rnd, max_rows * heads * dim, -1, 1);
    defer gpa.free(gate_h);

    var q_clip = try cuda.DeviceBuffer.fromHost(ctx.d, std.mem.sliceAsBytes(q_clip_h));
    defer q_clip.free();
    var rel_clip = try cuda.DeviceBuffer.fromHost(ctx.d, std.mem.sliceAsBytes(rel_clip_h));
    defer rel_clip.free();
    var proj = try cuda.DeviceBuffer.fromHost(ctx.d, std.mem.sliceAsBytes(proj_h));
    defer proj.free();
    const rig: Rig = .{ .gpa = gpa, .d = ctx.d, .s = stream, .kern = kern, .scratch = scratch, .q_clip = q_clip, .rel_clip = rel_clip, .proj = proj };
    try rig.put(scratch.r.sink, sink_h);
    try rig.put(scratch.r.gate, gate_h);

    var mem_a = try cuda.DeviceBuffer.alloc(ctx.d, 2 * la.cacheBytes(kv_heads));
    defer mem_a.free();
    var mem_b = try cuda.DeviceBuffer.alloc(ctx.d, 2 * lb.cacheBytes(kv_heads));
    defer mem_b.free();
    var mem_c = try cuda.DeviceBuffer.alloc(ctx.d, 2 * la.cacheBytes(kv_heads));
    defer mem_c.free();
    for ([_]cuda.DeviceBuffer{ mem_a, mem_b, mem_c }) |m| try m.fill8(0, stream.handle);
    try stream.synchronize();
    const ca = cachePair(la, mem_a.ptr);
    const cb = cachePair(lb, mem_b.ptr);
    const cc = cachePair(la, mem_c.ptr);

    var raw = try Raw.init(gpa, max_rows);
    defer raw.deinit(gpa);
    var p: usize = 0;
    while (p < filled) : (p += max_rows) {
        const rows = @min(max_rows, filled - p);
        raw.fill(rnd, rows, p == 0);
        _ = try rig.prep(raw, p, rows, 1, la, ca);
        _ = try rig.prep(raw, p, rows, 1, lb, cb);
    }
    std.debug.print("filled {d} positions of {d} (ring {d}), bf16 and 8-bit\n", .{ filled, context, la.ring });

    // the cases' own rows, in both caches, before C is built from B
    var case_raw: [cases.len]Raw = undefined;
    var made: usize = 0;
    defer for (case_raw[0..made]) |r| r.deinit(gpa);
    for (cases, &case_raw) |c, *r| {
        var cp = std.Random.DefaultPrng.init(c.seed);
        r.* = try Raw.init(gpa, max_rows);
        made += 1;
        r.fill(cp.random(), c.rows, false);
        _ = try rig.prep(r.*, c.p0, c.rows, c.streams, la, ca);
        _ = try rig.prep(r.*, c.p0, c.rows, c.streams, lb, cb);
    }

    var failures: usize = 0;
    const a_k = try downloadBf16(gpa, ctx.d, ca.k, la);
    defer gpa.free(a_k);
    const a_v = try downloadBf16(gpa, ctx.d, ca.v, la);
    defer gpa.free(a_v);
    failures += try checkQuantised(gpa, ctx.d, a_k, cb.k, lb, "K");
    failures += try checkQuantised(gpa, ctx.d, a_v, cb.v, lb, "V");
    const c_k = try downloadBf16(gpa, ctx.d, cb.k, lb);
    defer gpa.free(c_k);
    const c_v = try downloadBf16(gpa, ctx.d, cb.v, lb);
    defer gpa.free(c_v);
    const view_k: cuda.DeviceBuffer = .{ .d = ctx.d, .ptr = cc.k, .len = c_k.len * 2, .borrowed = true };
    try view_k.upload(0, std.mem.sliceAsBytes(c_k));
    const view_v: cuda.DeviceBuffer = .{ .d = ctx.d, .ptr = cc.v, .len = c_v.len * 2, .borrowed = true };
    try view_v.upload(0, std.mem.sliceAsBytes(c_v));

    const out_a = try gpa.alloc(u16, max_rows * heads * dim);
    defer gpa.free(out_a);
    const out_b = try gpa.alloc(u16, max_rows * heads * dim);
    defer gpa.free(out_b);
    const out_c = try gpa.alloc(u16, max_rows * heads * dim);
    defer gpa.free(out_c);
    var v_norms: [kv_heads]f64 = undefined;
    for (&v_norms, 0..) |*vn, h| {
        var sum: f64 = 0;
        for (0..filled) |pos| {
            var sq: f64 = 0;
            for (a_v[(h * la.ring + pos) * dim ..][0..dim]) |u| sq += @as(f64, widen(u)) * widen(u);
            sum += @sqrt(sq);
        }
        vn.* = sum / filled;
    }
    for (cases, case_raw) |c, r| {
        const n = c.rows * heads * dim;
        const call = try rig.prep(r, c.p0, c.rows, c.streams, lb, cb);
        try rig.attend(call, lb, cb, out_b[0..n]);
        try rig.attend(call, la, cc, out_c[0..n]);
        try rig.attend(call, la, ca, out_a[0..n]);
        const same = std.mem.eql(u16, out_b[0..n], out_c[0..n]);
        var worst: f64 = 0;
        var finite = true;
        for (0..c.rows * heads) |rh| {
            const vnorm = v_norms[rh % heads / group];
            var diff: f64 = 0;
            var norm: f64 = 0;
            for (out_a[rh * dim ..][0..dim], out_b[rh * dim ..][0..dim]) |a, b| {
                const fa = widen(a);
                const fb = widen(b);
                if (!std.math.isFinite(fb)) finite = false;
                diff += (fa - fb) * (fa - fb);
                norm += fa * fa;
            }
            worst = @max(worst, @sqrt(diff) / @max(@sqrt(norm), vnorm));
        }
        const ok = same and finite and worst <= max_rel_err;
        if (!ok) failures += 1;
        std.debug.print("{s}: 8-bit == widened bf16: {}, finite: {}, worst error vs bf16 cache, relative to V scale: {d:.5} (limit {d}) -> {s}\n", .{ c.name, same, finite, worst, max_rel_err, if (ok) "ok" else "FAIL" });
    }
    std.debug.print("{{\"verdict\": \"{s}\", \"failures\": {d}}}\n", .{ if (failures == 0) "CONFIRMED" else "BLOCKED", failures });
    return if (failures == 0) 0 else 1;
}
