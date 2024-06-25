value: f64 = undefined,
deriv: [2]f64 = undefined, // dη/dσ, dη/dγ

const Eta0: comptime_float = 0.0; // η₀
const Eta1: comptime_float = 1.36603; // η₁
const Eta2: comptime_float = 0.47719; // η₂
const Eta3: comptime_float = 0.11116; // η₃

pub fn update(self: *@This(), Gamma: *PseudoVoigtGamma, GammaL: f64) void {
    const alpha: f64 = GammaL / Gamma.value;

    var eta: f64 = Eta3; // η = η₀ + η₁α + η₂α² + η₃α³
    var beta: f64 = 0.0; // β = dη/dα = η₁ + 2η₂α + 3η₃α²

    inline for (.{ Eta2, Eta1, Eta0 }) |coeff| {
        beta = beta * alpha + eta;
        eta = eta * alpha + coeff;
    }

    self.value = eta;

    const temp: f64 = alpha * beta / GammaL;
    self.deriv[0] = -alpha * temp * Gamma.deriv[0];
    self.deriv[1] = temp * (2.0 - alpha * Gamma.deriv[1]);
}

const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
