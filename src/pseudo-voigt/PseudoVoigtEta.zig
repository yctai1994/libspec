value: f64 = undefined,
deriv: [2]f64 = undefined, // dη/dσ, dη/dγ

gamma: *PseudoVoigtGamma = undefined,
lorentz: *LorentzFWHM = undefined,

const Eta0: comptime_float = 0.0; // η₀
const Eta1: comptime_float = 1.36603; // η₁
const Eta2: comptime_float = -0.47719; // η₂
const Eta3: comptime_float = 0.11116; // η₃

const Self: type = @This();

pub fn init(allocator: mem.Allocator) !*Self {
    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.gamma = try PseudoVoigtGamma.init(allocator);
    self.lorentz = self.gamma.lorentz;

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.gamma.deinit(allocator);
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;
    const this: *Self = try Self.init(page);
    defer this.deinit(page);
}

pub fn forward(self: *Self, sigma: f64, gamma: f64) void {
    self.gamma.forward(sigma, gamma);

    const alpha: f64 = self.lorentz.value / self.gamma.value;

    var eta: f64 = Eta3; // η = η₀ + η₁α + η₂α² + η₃α³

    inline for (.{ Eta2, Eta1, Eta0 }) |coeff| {
        eta = eta * alpha + coeff;
    }

    self.value = eta;

    return;
}

pub fn update(self: *Self, Gamma: *PseudoVoigtGamma, GammaL: f64) void {
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

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
const LorentzFWHM = @import("./LorentzFWHM.zig");
