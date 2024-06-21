//! Probability Distributions/Profiles

const std = @import("std");
const math = std.math;

const pow2 = @import("../powers.zig").pow2;

const inv_sqrt_twopi: comptime_float = 1.0 / @sqrt(2.0 * math.pi);

pub const NormalDist = struct {
    mean: f64,
    stdd: f64, // standard deviation

    pub fn writeAll(self: *const NormalDist, parr: []f64, xarr: []f64) void {
        for (parr, xarr) |*p_i, x_i| {
            p_i.* = density(x_i, self.mean, self.stdd);
        }
        return;
    }
};

inline fn density(x: f64, m: f64, s: f64) f64 {
    return inv_sqrt_twopi * @exp(-0.5 * pow2(f64, (x - m) / s)) / s;
}
