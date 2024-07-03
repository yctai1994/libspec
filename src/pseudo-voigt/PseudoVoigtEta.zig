//! PseudoVoigtEta.zig
//! Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined,
deriv: f64 = undefined, // dPpV/dη
deriv_in: *f64 = undefined, // dy/dPpV
deriv_out: *f64 = undefined, // dy/dη

gamma: *PseudoVoigtGamma = undefined,
lorentz: *LorentzFWHM = undefined,

const Eta0: comptime_float = 0.0; // η₀
const Eta1: comptime_float = 1.36603; // η₁
const Eta2: comptime_float = -0.47719; // η₂
const Eta3: comptime_float = 0.11116; // η₃

const Self: type = @This();

fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.gamma = try PseudoVoigtGamma.init(allocator, tape);

    self.lorentz = self.gamma.lorentz;

    self.deriv = 1.0; // should be removed later
    self.deriv_in = &tape[0];
    self.deriv_out = &tape[5];

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.gamma.deinit(allocator);
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;

    var tape: []f64 = try page.alloc(f64, 9);
    defer page.free(tape);
    _ = &tape;

    const eta: *Self = try Self.init(page, tape);
    defer eta.deinit(page);
}

pub fn forward(self: *Self, sigma: f64, gamma: f64) void {
    self.gamma.forward(sigma, gamma);

    const alpha: f64 = self.lorentz.value / self.gamma.value;

    var eta: f64 = Eta3; // η = η₀ + η₁α + η₂α² + η₃α³
    var beta: f64 = 0.0; // β = dη/dα = η₁ + 2η₂α + 3η₃α²

    inline for (.{ Eta2, Eta1, Eta0 }) |coeff| {
        beta = beta * alpha + eta;
        eta = eta * alpha + coeff;
    }

    self.value = eta;

    beta /= self.gamma.value;

    self.gamma.deriv[2] = -alpha * beta; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
    self.lorentz.deriv[0] = beta; // [ dη/dΓL, dΓtot/dΓL ]

    return;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // final_deriv_out := [dy/dσ, dy/dγ]
    if (final_deriv_out.len != 2) unreachable;

    // (dy/dη) = (dPpV/dη) × (dy/dPpV)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    self.gamma.backward(final_deriv_out);

    return;
}

test "backward" {
    const page = testing.allocator;

    // Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
    var tape: []f64 = try page.alloc(f64, 9);
    defer page.free(tape);

    // deriv := [dy/dσ, dy/dγ]
    var deriv: []f64 = try page.alloc(f64, 2);
    defer page.free(deriv);
    _ = &deriv;

    for (0..9) |i| tape[i] = 1.0;

    const eta: *Self = try Self.init(page, tape);
    defer eta.deinit(page);

    const _sigma_: f64 = 2.171;
    const _gamma_: f64 = 1.305;

    eta.forward(_sigma_, _gamma_);
    eta.backward(deriv);

    std.debug.print("Eta        = {d} @ ({d}, {d})\n", .{ eta.value, _sigma_, _gamma_ });
    std.debug.print("dEta       = {d} @ ({d}, {d})\n", .{ deriv, _sigma_, _gamma_ });
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const LorentzFWHM = @import("./LorentzFWHM.zig");
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
