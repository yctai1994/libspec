mode: f64 = undefined,
scale: f64 = undefined,
value: f64 = undefined, // N( x | μ, Γ(σ, γ) )
deriv: [3]f64 = undefined, // dL/dμ, dL/dσ, dL/dγ

const inv_sqrt_twopi: comptime_float = 1.0 / @sqrt(2.0 * std.math.pi);
const inv_fwhm_factor: comptime_float = 1.0 / (2.0 * @sqrt(2.0 * @log(2.0)));

pub fn forward(self: *@This(), mu: f64, gamma: *PseudoVoigtGamma) void {
    self.mode = mu;
    self.scale = inv_fwhm_factor * gamma.value;
    return;
}

pub fn density(self: *@This(), x: f64) f64 {
    self.value = inv_sqrt_twopi * @exp(-0.5 * pow2((x - self.mode) / self.scale)) / self.scale;
    return self.value;
}

pub fn update(self: *@This(), Gamma: *PseudoVoigtGamma) void {
    const inv_Gamma: f64 = 1.0 / Gamma.value;
    const temp: f64 = self.value * (inv_Gamma - std.math.pi * self.value);

    self.deriv[1] = temp * Gamma.deriv[0];
    self.deriv[2] = temp * Gamma.deriv[1];

    return;
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
