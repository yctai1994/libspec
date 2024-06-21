const std = @import("std");

const findRoot = @import("./newton.zig").findRoot;
const NormalDist = @import("./prob-dist/normal.zig").NormalDist;

export fn c_find_root(
    fcall: *const fn (x: f64) callconv(.C) f64,
    deriv: *const fn (x: f64) callconv(.C) f64,
    target: f64,
    xleft: f64,
    xright: f64,
    ITMAX: usize, // Maximum allowed number of iterations.
) f64 {
    return findRoot(
        .C,
        fcall,
        deriv,
        target,
        xleft,
        xright,
        ITMAX,
    ) catch std.math.nan(f64);
}

export fn c_make_normal_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mean: f64,
    stdd: f64,
) void {
    const dist: NormalDist = .{ .mean = mean, .stdd = stdd };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}
