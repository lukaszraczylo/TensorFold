//! TF_INK2_PROF=count|time|visit_sync|dup_head: per decode token, launches, host syncs, copies and host time in the driver calls; `time`
//! adds CUDA events around every launch for GPU time per kernel, per visit and the idle gaps; `visit_sync` adds a stream
//! sync at every visit boundary and `dup_head` launches the head's matrix product twice (a known +4 ms per token, the
//! tuning loop's sensitivity control). Unset leaves the driver alone.
const std = @import("std");
const cuda = @import("cuda");
const sys = @import("sys");

const abi = cuda.abi;

const env_name = "TF_INK2_PROF";
const max_launches = 8192;
const max_names = 512;
const slots = 1024;
/// Visit tags 0..47 are schedule visits; these two cover the embedding and the head.
pub const tag_embed: u8 = 48;
pub const tag_head: u8 = 49;
const tags = 50;
const report_every = 32;
/// The decode token whose launch sequence is printed whole (it is the 6th timed one: caches are warm).
const seq_token = 6;

const Mode = enum { off, count, time, visit_sync, dup_head };

const Name = struct { handle: abi.Function = null, text: []const u8 = "", calls: u64 = 0, busy_us: f64 = 0, gap_us: f64 = 0 };

const Orig = struct {
    launch: @TypeOf(@as(abi.Api, undefined).cuLaunchKernel),
    launch_ex: @TypeOf(@as(abi.Api, undefined).cuLaunchKernelEx),
    sync: @TypeOf(@as(abi.Api, undefined).cuStreamSynchronize),
    htod: @TypeOf(@as(abi.Api, undefined).cuMemcpyHtoD_v2),
    htod_async: @TypeOf(@as(abi.Api, undefined).cuMemcpyHtoDAsync_v2),
    dtoh: @TypeOf(@as(abi.Api, undefined).cuMemcpyDtoH_v2),
    dtoh_async: @TypeOf(@as(abi.Api, undefined).cuMemcpyDtoHAsync_v2),
};

/// Sums of one window of tokens; `Token` is what the current token has added so far.
const Sums = struct {
    tokens: u64 = 0,
    wall_ns: u64 = 0,
    launches: u64 = 0,
    launch_ns: u64 = 0,
    syncs: u64 = 0,
    sync_ns: u64 = 0,
    htod: u64 = 0,
    htod_bytes: u64 = 0,
    htod_ns: u64 = 0,
    dtoh: u64 = 0,
    dtoh_bytes: u64 = 0,
    dtoh_ns: u64 = 0,
    busy_us: f64 = 0,
    gap_us: f64 = 0,
};

var mode: Mode = .off;
var active = false;
var tag: u8 = tag_embed;
var stream_h: abi.Stream = null;
var head_dup_done = false;
var tokens_seen: u64 = 0;
var drv: *const cuda.Driver = undefined;
var orig: Orig = undefined;
var names: [max_names]Name = @splat(.{});
var n_names: usize = 0;
var slot_of: [slots]u16 = @splat(0);
var sums: Sums = .{};
var cur: Sums = .{};
var t_begin: u64 = 0;
var n: usize = 0;
var events: [2 * max_launches + 2]abi.Event = undefined;
var launch_name: [max_launches]u16 = undefined;
var launch_tag: [max_launches]u8 = undefined;
var vis_busy_us: [tags]f64 = @splat(0);
var vis_gap_us: [tags]f64 = @splat(0);
var vis_launches: [tags]u64 = @splat(0);

fn now() u64 {
    var t: sys.c.timespec = undefined;
    _ = sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &t);
    return @intCast(@as(i128, t.sec) * std.time.ns_per_s + t.nsec);
}

fn nameOf(f: abi.Function) u16 {
    var i: usize = (@intFromPtr(f) >> 4) & (slots - 1);
    while (slot_of[i] != 0) : (i = (i + 1) & (slots - 1)) {
        if (names[slot_of[i] - 1].handle == f) return slot_of[i] - 1;
    }
    if (n_names == max_names) return 0;
    names[n_names] = .{ .handle = f, .text = drv.functionName(f) };
    n_names += 1;
    slot_of[i] = @intCast(n_names);
    return @intCast(n_names - 1);
}

fn timed() bool {
    return active and mode == .time;
}

fn record(stream: abi.Stream, idx: usize) void {
    _ = drv.api.cuEventRecord(events[idx], stream);
}

fn count(f: abi.Function, stream: abi.Stream) void {
    cur.launches += 1;
    vis_launches[tag] += 1;
    if (timed() and n < max_launches) {
        launch_name[n] = nameOf(f);
        launch_tag[n] = tag;
        record(stream, 1 + 2 * n);
    }
}

fn after(stream: abi.Stream, t0: u64) void {
    cur.launch_ns += now() - t0;
    if (timed() and n < max_launches) {
        record(stream, 2 + 2 * n);
        n += 1;
    }
}

fn launchThunk(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, params: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
    if (!active) return orig.launch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    if (mode == .dup_head and tag == tag_head and !head_dup_done) {
        head_dup_done = true; // the head's first launch is the logits product: it rewrites its output, so a repeat is harmless
        _ = orig.launch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    }
    count(f, stream);
    const t0 = now();
    const r = orig.launch(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    after(stream, t0);
    return r;
}

fn launchExThunk(cfg: *const abi.LaunchConfig, f: abi.Function, params: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
    if (!active) return orig.launch_ex(cfg, f, params, extra);
    count(f, cfg.stream);
    const t0 = now();
    const r = orig.launch_ex(cfg, f, params, extra);
    after(cfg.stream, t0);
    return r;
}

fn syncThunk(s: abi.Stream) callconv(.c) abi.Result {
    if (!active) return orig.sync(s);
    const t0 = now();
    const r = orig.sync(s);
    cur.syncs += 1;
    cur.sync_ns += now() - t0;
    return r;
}

fn htodThunk(d: abi.DevicePtr, src: ?*const anyopaque, bytes: usize) callconv(.c) abi.Result {
    if (!active) return orig.htod(d, src, bytes);
    const t0 = now();
    const r = orig.htod(d, src, bytes);
    cur.htod += 1;
    cur.htod_bytes += bytes;
    cur.htod_ns += now() - t0;
    return r;
}

fn htodAsyncThunk(d: abi.DevicePtr, src: ?*const anyopaque, bytes: usize, s: abi.Stream) callconv(.c) abi.Result {
    if (!active) return orig.htod_async(d, src, bytes, s);
    const t0 = now();
    const r = orig.htod_async(d, src, bytes, s);
    cur.htod += 1;
    cur.htod_bytes += bytes;
    cur.htod_ns += now() - t0;
    return r;
}

fn dtohThunk(dst: ?*anyopaque, src: abi.DevicePtr, bytes: usize) callconv(.c) abi.Result {
    if (!active) return orig.dtoh(dst, src, bytes);
    const t0 = now();
    const r = orig.dtoh(dst, src, bytes);
    cur.dtoh += 1;
    cur.dtoh_bytes += bytes;
    cur.dtoh_ns += now() - t0;
    return r;
}

fn dtohAsyncThunk(dst: ?*anyopaque, src: abi.DevicePtr, bytes: usize, s: abi.Stream) callconv(.c) abi.Result {
    if (!active) return orig.dtoh_async(dst, src, bytes, s);
    const t0 = now();
    const r = orig.dtoh_async(dst, src, bytes, s);
    cur.dtoh += 1;
    cur.dtoh_bytes += bytes;
    cur.dtoh_ns += now() - t0;
    return r;
}

/// Wraps the driver's launch, sync and copy entry points when TF_INK2_PROF is `count` or `time`.
pub fn install(d: *const cuda.Driver) !void {
    const raw = sys.getenv(env_name) orelse return;
    mode = std.meta.stringToEnum(Mode, raw) orelse return;
    if (mode == .off) return;
    drv = d;
    const api: *abi.Api = @constCast(&d.api);
    orig = .{ .launch = api.cuLaunchKernel, .launch_ex = api.cuLaunchKernelEx, .sync = api.cuStreamSynchronize, .htod = api.cuMemcpyHtoD_v2, .htod_async = api.cuMemcpyHtoDAsync_v2, .dtoh = api.cuMemcpyDtoH_v2, .dtoh_async = api.cuMemcpyDtoHAsync_v2 };
    if (mode == .time) for (&events) |*e| try d.check(d.api.cuEventCreate(e, 0), "cuEventCreate");
    api.cuLaunchKernel = launchThunk;
    api.cuLaunchKernelEx = launchExThunk;
    api.cuStreamSynchronize = syncThunk;
    api.cuMemcpyHtoD_v2 = htodThunk;
    api.cuMemcpyHtoDAsync_v2 = htodAsyncThunk;
    api.cuMemcpyDtoH_v2 = dtohThunk;
    api.cuMemcpyDtoHAsync_v2 = dtohAsyncThunk;
    std.log.info("{s}={s}: per-token launch/sync/copy counts every {d} decode tokens", .{ env_name, raw, report_every });
}

pub fn enabled() bool {
    return mode != .off;
}

/// Marks the schedule visit (or tag_embed / tag_head) the next launches belong to.
pub fn at(t: u8) void {
    if (active and mode == .visit_sync and t != tag) _ = orig.sync(stream_h);
    tag = t;
}

/// The start of a decode token (before its embedding); `stream` takes the timeline's origin event.
pub fn begin(stream: abi.Stream) void {
    if (mode == .off) return;
    cur = .{};
    n = 0;
    tag = tag_embed;
    stream_h = stream;
    head_dup_done = false;
    active = true;
    if (mode == .time) record(stream, 0);
    t_begin = now();
}

/// The end of a decode token, after the host read its pick (the stream is idle: every event has fired).
pub fn end() void {
    if (!active) return;
    const wall = now() - t_begin;
    active = false;
    tokens_seen += 1;
    cur.wall_ns = wall;
    if (mode == .time) resolve();
    if (mode == .time and tokens_seen == seq_token) dumpSequence();
    add();
    if (sums.tokens >= report_every) report();
}

fn elapsedUs(a: usize, b: usize) f64 {
    var msec: f32 = 0;
    _ = drv.api.cuEventElapsedTime(&msec, events[a], events[b]);
    return @as(f64, msec) * 1000.0;
}

fn resolve() void {
    var prev: usize = 0;
    for (0..n) |i| {
        const gap = elapsedUs(prev, 1 + 2 * i);
        const busy = elapsedUs(1 + 2 * i, 2 + 2 * i);
        prev = 2 + 2 * i;
        const nm = &names[launch_name[i]];
        nm.calls += 1;
        nm.busy_us += busy;
        nm.gap_us += gap;
        vis_busy_us[launch_tag[i]] += busy;
        vis_gap_us[launch_tag[i]] += gap;
        cur.busy_us += busy;
        cur.gap_us += gap;
    }
}

fn dumpSequence() void {
    var prev: usize = 0;
    std.debug.print("ink2prof seq-begin launches={d}\n", .{n});
    for (0..n) |i| {
        std.debug.print("ink2prof seq {d} {s} busy_us={d:.2} gap_us={d:.2}\n", .{ launch_tag[i], names[launch_name[i]].text, elapsedUs(1 + 2 * i, 2 + 2 * i), elapsedUs(prev, 1 + 2 * i) });
        prev = 2 + 2 * i;
    }
}

fn add() void {
    inline for (@typeInfo(Sums).@"struct".field_names) |name| @field(sums, name) += @field(cur, name);
    sums.tokens += 1;
}

fn ms(ns: u64, tok: f64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6 / tok;
}

fn report() void {
    const tok: f64 = @floatFromInt(sums.tokens);
    const s = sums;
    std.debug.print(
        "ink2prof agg mode={s} tokens={d} wall_ms={d:.3} launches={d:.1} launch_call_ms={d:.3} syncs={d:.2} sync_wait_ms={d:.3} htod={d:.2} htod_kb={d:.2} htod_ms={d:.3} dtoh={d:.2} dtoh_ms={d:.3} busy_ms={d:.3} gap_ms={d:.3}\n",
        .{ @tagName(mode), s.tokens, ms(s.wall_ns, tok), @as(f64, @floatFromInt(s.launches)) / tok, ms(s.launch_ns, tok), @as(f64, @floatFromInt(s.syncs)) / tok, ms(s.sync_ns, tok), @as(f64, @floatFromInt(s.htod)) / tok, @as(f64, @floatFromInt(s.htod_bytes)) / 1024 / tok, ms(s.htod_ns, tok), @as(f64, @floatFromInt(s.dtoh)) / tok, ms(s.dtoh_ns, tok), s.busy_us / 1000 / tok, s.gap_us / 1000 / tok },
    );
    if (mode == .time) {
        for (names[0..n_names]) |nm| {
            if (nm.calls == 0) continue;
            std.debug.print("ink2prof k {s} calls={d:.2} busy_us={d:.2} gap_us={d:.2}\n", .{ nm.text, @as(f64, @floatFromInt(nm.calls)) / tok, nm.busy_us / tok, nm.gap_us / tok });
        }
        for (0..tags) |t| {
            if (vis_launches[t] == 0) continue;
            std.debug.print("ink2prof v {d} launches={d:.1} busy_us={d:.2} gap_us={d:.2}\n", .{ t, @as(f64, @floatFromInt(vis_launches[t])) / tok, vis_busy_us[t] / tok, vis_gap_us[t] / tok });
        }
    }
    sums = .{};
    for (names[0..n_names]) |*nm| {
        nm.calls = 0;
        nm.busy_us = 0;
        nm.gap_us = 0;
    }
    vis_busy_us = @splat(0);
    vis_gap_us = @splat(0);
    vis_launches = @splat(0);
}

test "with TF_INK2_PROF unset every hook is inert" {
    try std.testing.expect(!enabled());
    begin(null);
    try std.testing.expect(!active);
    end();
    try std.testing.expectEqual(@as(u64, 0), tokens_seen);
}

test "profile modes are the documented words" {
    try std.testing.expectEqual(Mode.dup_head, std.meta.stringToEnum(Mode, "dup_head").?);
    try std.testing.expect(std.meta.stringToEnum(Mode, "1") == null);
}
