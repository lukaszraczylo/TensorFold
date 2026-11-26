//! The draft head a serving engine holds: the checkpoint's DSpark head (drafter.zig) or a DFlash2 draft model (dflash2.zig).
const std = @import("std");
const cuda = @import("cuda");
const dflash2 = @import("dflash2.zig");
const drafter = @import("drafter.zig");

pub const proposals = drafter.proposals;
comptime {
    std.debug.assert(dflash2.proposals == drafter.proposals and dflash2.write_rows == drafter.write_rows);
}

pub const Head = union(enum) {
    dspark: drafter.Drafter,
    dflash2: dflash2.Dflash2,

    pub fn deinit(h: *Head) void {
        switch (h.*) {
            inline else => |*x| x.deinit(),
        }
    }

    pub fn writeContext(h: *Head, s: cuda.Stream, stream: u32, taps: u64, rows: usize, pos0: u64) !void {
        switch (h.*) {
            inline else => |*x| try x.writeContext(s, stream, taps, rows, pos0),
        }
    }

    pub fn setRnnContext(h: *Head, s: cuda.Stream, stream: u32, taps: u64, row: usize) !void {
        switch (h.*) {
            inline else => |*x| try x.setRnnContext(s, stream, taps, row),
        }
    }

    pub fn propose(h: *Head, s: cuda.Stream, stream: u32, anchor: u32, pos: u64, out: *[proposals]u32, chain: *[proposals]f32) !void {
        switch (h.*) {
            inline else => |*x| try x.propose(s, stream, anchor, pos, out, chain),
        }
    }

    /// A kept prompt state of `at` positions is back in `stream`'s context.
    pub fn restored(h: *Head, stream: u32, at: u64) void {
        switch (h.*) {
            .dspark => {},
            .dflash2 => |*x| x.restored(stream, at),
        }
    }

    pub fn vcap(h: *const Head) bool {
        return switch (h.*) {
            inline else => |*x| x.vcap,
        };
    }

    pub fn vcapFloor(h: *const Head) ?f64 {
        return switch (h.*) {
            inline else => |*x| x.vcap_floor,
        };
    }

    pub fn snapBytes(h: *const Head, at: u32) u64 {
        return switch (h.*) {
            inline else => |*x| x.snapBytes(at),
        };
    }

    pub fn each(h: *const Head, stream: u32, at: u32, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: u64, len: u64) anyerror!void) !void {
        switch (h.*) {
            inline else => |*x| try x.each(stream, at, ctx, f),
        }
    }
};
