const std = @import("std");

const findRoot = @import("./newton.zig").findRoot;
const NormalDist = @import("./prob-dist/normal.zig").NormalDist;
const LorentzDist = @import("./prob-dist/lorentz.zig").LorentzDist;
const PseudoVoigtDist = @import("./prob-dist/pseudo-voigt.zig").PseudoVoigtDist;
const PseudoVoigtDeriv = @import("./prob-dist/pseudo-voigt-deriv.zig");

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
    mode: f64,
    scale: f64,
) void {
    const dist: NormalDist = .{ .mode = mode, .scale = scale };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}

export fn c_make_lorentz_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mode: f64,
    scale: f64,
) void {
    const dist: LorentzDist = .{ .mode = mode, .scale = scale };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}

export fn c_make_pvoigt_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mode: f64,
    scaleN: f64,
    scaleL: f64,
) void {
    const dist: PseudoVoigtDist = .{ .mode = mode, .scaleN = scaleN, .scaleL = scaleL };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}

export fn c_Gamma(Gamma_G: f64, Gamma_L: f64) f64 {
    return PseudoVoigtDeriv.Gamma(Gamma_G, Gamma_L);
}

export fn c_dGamma(Gamma_G: f64, Gamma_L: f64, Gtot: *f64, dGds: *f64, dGdg: *f64) void {
    return PseudoVoigtDeriv.dGamma(Gamma_G, Gamma_L, Gtot, dGds, dGdg);
}
