//! Inkling 2 Mini's kept states for the disk store (native/cuda_kept.zig): layout, a blank one, what they are read under.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const be = @import("backend.zig");
const snap = @import("snap.zig");

const Allocator = std.mem.Allocator;

pub const Target = be.Inkling;
pub const State = snap.Saved;
pub const probe_seed: u64 = 0x6932;

pub fn driver(t: *const Target) *const cuda.Driver {
    return t.c.d;
}

fn draftBytes(t: *const Target, at: u32) u64 {
    return if (t.draftHead()) |h| h.snapBytes(at) else 0;
}

/// The global planes (rows from position 0), the sliding tails, the conv states, then the draft head's context.
pub fn parts(t: *const Target, at: u32, out: anytype, a: Allocator) !void {
    try snap.parts(t.c.cache, &t.c.sched, at, draftBytes(t, at), out, a);
}

pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
    return .{ st.buf, st.draft };
}

pub fn blank(t: *Target, at: u32) !*State {
    return t.blankState(at);
}

/// A state without the head's context cannot serve a stream whose head is on.
pub fn whole(t: *const Target, st: *const State) bool {
    return t.mtp == null or st.draft != null;
}

pub fn drafts(_: *const Target) bool {
    return false; // the head absorbs every row it sees; the probe's state holds its context either way
}

pub fn context(t: *const Target) usize {
    return t.context;
}

pub fn save(t: *Target, s: *lanes.Stream, at: u32) !*State {
    return t.saveState(s, at);
}

pub fn drop(t: *Target, st: *State) void {
    t.dropState(st);
}

/// What the image's bytes depend on past the probe's hash: the cache form, the chunk grid, the head and its rings.
pub fn facts(t: *const Target) [4]u64 {
    var h = std.hash.Wyhash.init(0x6932);
    if (t.draftHead()) |d| for (d.layers[0..d.m.layers]) |l| h.update(std.mem.asBytes(&@as(u64, l.ring)));
    const q8 = for (0..t.c.sched.n) |i| {
        if (t.c.sched.visits[i].global) break t.c.cache.layerAt(@intCast(i)).q8;
    } else false;
    return .{ @intFromBool(q8), t.c.scratch.geometry.max_rows, @intFromBool(t.mtp != null), h.final() };
}
