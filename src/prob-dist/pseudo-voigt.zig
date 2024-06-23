//! Pseudo-Voigt Profile/Distribution
const std = @import("std");
const math = std.math;

const poly = @import("../poly.zig");
const normal = @import("./normal.zig");
const lorentz = @import("./lorentz.zig");
const nthRoot = @import("../nth-root.zig").nthRoot;

pub const PseudoVoigtDist = struct {
    mode: f64, // the value that most often appears
    scaleN: f64, // statistical dispersion of the Normal dist.
    scaleL: f64, // statistical dispersion of the Lorentz dist.

    pub fn writeAll(self: *const PseudoVoigtDist, parr: []f64, xarr: []f64) void {
        var fwhmN: f64 = normal.getFWHM(self.scaleN);
        var fwhmL: f64 = lorentz.getFWHM(self.scaleL);

        const fwhm: f64 = getFWHM(fwhmN, fwhmL);
        const eta: f64 = poly.evalPolySum(f64, 3, fwhmL / fwhm, .{ 0.11116, -0.47719, 1.36603, 0.0 });

        fwhmN = normal.getScale(fwhm);
        fwhmL = lorentz.getScale(fwhm);

        for (parr, xarr) |*p_i, x_i| {
            p_i.* = eta * lorentz.density(x_i, self.mode, fwhmL) + (1.0 - eta) * normal.density(x_i, self.mode, fwhmN);
        }
        return;
    }
};

fn getFWHM(fwhmN: f64, fwhmL: f64) f64 {
    var coeff: [5 + 1]f64 = .{ 1.0, 0.07842, 4.47163, 2.42843, 2.69269, 1.0 };
    poly.evalPolyArray(f64, 5, fwhmN, &coeff);
    return nthRoot(f64, 5, poly.evalPolySum(f64, 5, fwhmL, coeff)) catch unreachable;
}
