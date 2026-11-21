//! Inkling 2's CUDA family in the build: its module, its check tools and its host regressions.
const std = @import("std");

/// The weight map's `safetensors` import, served from the core's copy: one file cannot sit in two modules.
fn shim(b: *std.Build) std.Build.LazyPath {
    const text =
        \\const st = @import("core").safetensors;
        \\pub const DType = st.DType;
        \\pub const Entry = st.Entry;
        \\pub const Header = st.Header;
        \\pub const headerSize = st.headerSize;
        \\pub const parseHeader = st.parseHeader;
        \\
    ;
    return b.addWriteFiles().add("inkling2_safetensors.zig", text);
}

/// The Inkling 2 CUDA family over the runtime, the core and the lane core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/inkling2.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("safetensors", b.createModule(.{ .root_source_file = shim(b), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} }));
    return m;
}

/// The family in tf-cuda-test's `runner`, where the kernel contract reaches its launchers.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, runner: *std.Build.Module) void {
    _ = strip;
    runner.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
}

/// The family's host regressions (config, chat wiring, reply split, admission, the attention plan) in `all`.
pub fn hostTests(b: *std.Build, host: std.Build.ResolvedTarget, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, all: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, core, lanes) }));
    b.step("test-inkling2-cuda-host", "Inkling 2's CUDA family host regressions (no GPU work)").dependOn(&run.step);
    all.dependOn(&run.step);
}

/// `tf-ink2-check`: Inkling 2 Mini's CUDA rows for the reference gates.
pub fn checkTool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/check_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    root.addImport("cuda", cuda);
    root.addImport("inkling2", module(b, target, optimize, cuda, core, lanes));
    const install = b.addInstallArtifact(b.addExecutable(.{ .name = "tf-ink2-check", .root_module = root }), .{});
    b.getInstallStep().dependOn(&install.step);
    b.step("ink2-check", "Build the Inkling 2 Mini CUDA check tool").dependOn(&install.step);
}
