mode: f64 = undefined,
scale: f64 = undefined,
value: f64 = undefined, // L( x | μ, Γ(σ, γ) )
deriv: [3]f64 = undefined, // dL/dμ, dL/dσ, dL/dγ

gamma: *PseudoVoigtGamma = undefined,

pub fn forward(self: *@This(), mu: f64) void {
    self.mode = mu;
    self.scale = 0.5 * self.gamma.value;
    return;
}

pub fn density(self: *@This(), x: f64) f64 {
    self.value = 1.0 / (std.math.pi * self.scale * (1.0 + pow2((x - self.mode) / self.scale)));
    return self.value;
}

pub fn backward(self: *@This(), x: f64, deriv: []f64) void {
    if (deriv.len != 3) unreachable;

    const prob: f64 = self.density(x);
    const arg1: f64 = prob / self.scale;
    const arg2: f64 = 2.0 * math.pi * pow2(prob);
    const temp: f64 = 0.5 * (arg1 - arg2);

    deriv[0] = (x - self.mode) * arg2 / self.scale; // dL/dμ
    deriv[1] = temp * self.gamma.deriv[0]; // dL/dσ
    deriv[2] = temp * self.gamma.deriv[1]; // dL/dγ

    return;
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const math = std.math;
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
