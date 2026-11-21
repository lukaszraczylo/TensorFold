//! Inkling 2 Mini's loop kernels (inkling2/loop.cu): the image's module and its functions.
const cuda = @import("cuda");

pub const Kernels = struct {
    mod: cuda.Module,
    fill: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .fill = try mod.function("tf_ink2l_fill"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};
