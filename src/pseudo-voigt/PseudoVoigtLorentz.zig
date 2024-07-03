//! PseudoVoigtLorentz.zig
//! Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // L( x | μ, Γ(σ, γ) )
deriv: f64 = undefined, // dPpV/dPL
deriv_in: *f64 = undefined, // dy/dPpV
deriv_out: *f64 = undefined, // dy/dPL

mode: *PseudoVoigtMode = undefined,
scale: *PseudoLorentzScale = undefined,

const Self: type = @This();

fn init(allocator: mem.Allocator, tape: []f64, gamma: *PseudoVoigtGamma) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.mode = try PseudoVoigtMode.init(allocator, tape);
    errdefer self.mode.deinit(allocator);

    self.scale = try PseudoLorentzScale.init(allocator, tape, gamma);

    self.deriv_in = &tape[0];
    self.deriv_out = &tape[2];

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.mode.deinit(allocator);
    self.scale.deinit(allocator);
    allocator.destroy(self);
    return;
}

pub fn forward(self: *@This(), x: f64) void {
    self.scale.forward();

    const prob: f64 = density(x, self.mode.value, self.scale.value);
    const arg1: f64 = prob / self.scale.value;
    const arg2: f64 = 2.0 * math.pi * pow2(prob);

    self.value = prob;
    self.mode.deriv[1] = (x - self.mode.value) * arg2 / self.scale.value; // [ dPG/dμ, dPL/dμ ]
    self.scale.deriv = arg1 - arg2; // dPL/dγV

    return;
}

inline fn density(x: f64, mu: f64, gamma: f64) f64 {
    return 1.0 / (math.pi * gamma * (1.0 + pow2((x - mu) / gamma)));
}

pub fn backward(self: *Self) void {
    // (dy/dPL) = (dPpV/dPL) × (dy/dPpV)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    self.scale.backward();

    return;
}

test "backward: y ≡ PL" {
    const page = testing.allocator;

    var tape: []f64 = try page.alloc(f64, 9);
    defer page.free(tape);

    @memset(tape, 1.0);
    _ = &tape;

    // deriv := [dy/dμ, dy/dσ, dy/dγ]
    var deriv: []f64 = try page.alloc(f64, 3);
    defer page.free(deriv);
    _ = &deriv;

    const gamma: *PseudoVoigtGamma = try PseudoVoigtGamma.init(page, tape);
    defer gamma.deinit(page);

    const lorentz: *Self = try Self.init(page, tape, gamma);
    defer lorentz.deinit(page);

    const _x_: f64 = 1.213;
    const _mode_: f64 = 0.878;
    const _sigma_: f64 = 2.171;
    const _gamma_: f64 = 1.305;

    lorentz.mode.forward(_mode_); // should be improved
    gamma.forward(_sigma_, _gamma_);
    lorentz.forward(_x_);

    // Produce dy/dPL = dPL/dPL = 1
    lorentz.deriv = 1.0;

    lorentz.backward();
    lorentz.mode.backward(&deriv[0]); // should be improved

    gamma.backward(deriv[1..]);

    std.debug.print(
        "PL         = {d} @ ({d}, {d}, {d}, {d})\n",
        .{ lorentz.value, _x_, _mode_, _sigma_, _gamma_ },
    );
    std.debug.print(
        "dPL        = {d} @ ({d}, {d}, {d}, {d})\n",
        .{ deriv, _x_, _mode_, _sigma_, _gamma_ },
    );
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;

const PseudoVoigtMode = @import("./PseudoVoigtMode.zig");
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
const PseudoLorentzScale = @import("./PseudoLorentzScale.zig");
