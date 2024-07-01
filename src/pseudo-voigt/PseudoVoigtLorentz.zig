mode: f64 = undefined,
scale: f64 = undefined,
value: f64 = undefined, // L( x | μ, Γ(σ, γ) )
deriv: [3]f64 = undefined, // dL/dμ, dL/dσ, dL/dγ

pub fn forward(self: *@This(), mu: f64, gamma: *PseudoVoigtGamma) void {
    self.mode = mu;
    self.scale = 0.5 * gamma.value;
    return;
}

pub fn density(self: *@This(), x: f64) f64 {
    self.value = 1.0 / (std.math.pi * self.scale * (1.0 + pow2((x - self.mode) / self.scale)));
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
