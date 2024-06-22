//! Cauchy–Lorentz Profile/Distribution
const std = @import("std");
const math = std.math;

const pow2 = @import("../powers.zig").pow2;

pub const LorentzDist = struct {
    mode: f64, // the value that most often appears
    scale: f64, // statistical dispersion

    pub fn writeAll(self: *const LorentzDist, parr: []f64, xarr: []f64) void {
        for (parr, xarr) |*p_i, x_i| {
            p_i.* = density(x_i, self.mode, self.scale);
        }
        return;
    }
};

inline fn density(x: f64, m: f64, s: f64) f64 {
    return 1.0 / (math.pi * s * (1.0 + pow2(f64, (x - m) / s)));
}
