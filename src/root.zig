const std = @import("std");

const findRoot = @import("./newton.zig").findRoot;
const NormalDist = @import("./prob-dist/normal.zig").NormalDist;
const LorentzDist = @import("./prob-dist/lorentz.zig").LorentzDist;
const PseudoVoigtDist = @import("./prob-dist/pseudo-voigt.zig").PseudoVoigtDist;
const PseudoVoigtGamma = @import("./prob-dist/PseudoVoigtGamma.zig");
const PseudoVoigtEta = @import("./prob-dist/PseudoVoigtEta.zig");

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

export fn c_dGamma(GammaG: f64, GammaL: f64, Gtot: *f64, dGds: *f64, dGdg: *f64) void {
    var pseudo_voigt_gamma: PseudoVoigtGamma = .{};
    pseudo_voigt_gamma.update(GammaG, GammaL);

    Gtot.* = pseudo_voigt_gamma.value;
    dGds.* = pseudo_voigt_gamma.deriv[0];
    dGdg.* = pseudo_voigt_gamma.deriv[1];

    return;
}

export fn c_dEta(GammaG: f64, GammaL: f64, Etot: *f64, dEds: *f64, dEdg: *f64) void {
    var pseudo_voigt_gamma: PseudoVoigtGamma = .{};
    pseudo_voigt_gamma.update(GammaG, GammaL);

    var pseudo_voigt_eta: PseudoVoigtEta = .{};
    pseudo_voigt_eta.update(&pseudo_voigt_gamma, GammaL);

    Etot.* = pseudo_voigt_eta.value;
    dEds.* = pseudo_voigt_eta.deriv[0];
    dEdg.* = pseudo_voigt_eta.deriv[1];

    return;
}
