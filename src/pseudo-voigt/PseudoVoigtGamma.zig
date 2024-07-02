// Tape: [..., dy/dσV, dy/dγV, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined,
deriv: [2]f64 = undefined, // [ dσV/dΓtot, dγV/dΓtot ]
deriv_in: []f64 = undefined, // [ dy/dσV, dy/dγV ]
deriv_out: *f64 = undefined, // dy/dΓtot

normal: *NormalFWHM = undefined,
lorentz: *LorentzFWHM = undefined,

const G0: comptime_float = 1.0;
const G1: comptime_float = 0.07842;
const G2: comptime_float = 4.47163;
const G3: comptime_float = 2.42843;
const G4: comptime_float = 2.69269;
const G5: comptime_float = 1.0;

const Self: type = @This();

fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 5) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.normal = try NormalFWHM.init(allocator, tape[2..]);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try LorentzFWHM.init(allocator, tape[2..]);

    self.deriv = .{ 0.5, 0.5 }; // should be removed later
    self.deriv_in = tape[0..2];
    self.deriv_out = &tape[2];

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.normal.deinit(allocator);
    self.lorentz.deinit(allocator);
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;

    var tape: []f64 = try page.alloc(f64, 5);
    defer page.free(tape);
    _ = &tape;

    const gamma: *Self = try Self.init(page, tape);
    defer gamma.deinit(page);
}

pub fn forward(self: *Self, sigma: f64, gamma: f64) void {
    self.normal.forward(sigma);
    self.lorentz.forward(gamma);

    const GammaG: f64 = self.normal.value;
    const GammaL: f64 = self.lorentz.value;

    var cGtot: [6]f64 = comptime .{ G0, G1, G2, G3, G4, G5 };
    var cdGdG: [5]f64 = comptime .{ G1 * 1.0, G2 * 2.0, G3 * 3.0, G4 * 4.0, G5 * 5.0 };
    var cdGdL: [5]f64 = comptime .{ G0 * 5.0, G1 * 4.0, G2 * 3.0, G3 * 2.0, G4 * 1.0 };

    var temp: f64 = GammaG;
    for (cGtot[1..5], cdGdG[1..], cdGdL[1..]) |*c_Gtot, *c_dGdG, *c_dGdL| {
        c_Gtot.* *= temp;
        c_dGdG.* *= temp;
        c_dGdL.* *= temp;
        temp *= GammaG;
    }
    cGtot[5] *= temp;

    var Gtot: f64 = cGtot[0];
    var dGdG: f64 = cdGdG[0];
    var dGdL: f64 = cdGdL[0];

    for (cGtot[1..5], cdGdG[1..], cdGdL[1..]) |c_Gtot, c_dGdG, c_dGdL| {
        Gtot = Gtot * GammaL + c_Gtot;
        dGdG = dGdG * GammaL + c_dGdG;
        dGdL = dGdL * GammaL + c_dGdL;
    }
    Gtot = Gtot * GammaL + cGtot[5];

    temp = @log(Gtot);
    Gtot = @exp(0.2 * temp);
    temp = @exp(-0.8 * temp) * 0.2;

    self.value = Gtot;
    self.normal.deriv = temp * dGdG; // dΓtot/dΓG
    self.lorentz.deriv = temp * dGdL; // dΓtot/dΓL

    return;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // final_deriv_out := [dy/dσ, dy/dγ]
    if (final_deriv_out.len != 2) unreachable;

    // (dy/dΓtot) = [dσV/dΓtot, dγV/dΓtot]ᵀ ⋅ [dy/dσV, dy/dγV]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| {
        temp += deriv * deriv_in;
    }

    self.deriv_out.* = temp;
    self.normal.backward(&final_deriv_out[0]);
    self.lorentz.backward(&final_deriv_out[1]);

    return;
}

test "backward" {
    const page = testing.allocator;

    // Tape: [dy/dσV, dy/dγV, dy/dΓtot, dy/dΓG, dy/dΓL]
    var tape: []f64 = try page.alloc(f64, 5);
    defer page.free(tape);

    // deriv := [dy/dσ, dy/dγ]
    var deriv: []f64 = try page.alloc(f64, 2);
    defer page.free(deriv);
    _ = &deriv;

    tape[0] = 1.0;
    tape[1] = 1.0;

    const gamma: *Self = try Self.init(page, tape);
    defer gamma.deinit(page);

    gamma.forward(2.171, 1.305);
    gamma.backward(deriv);

    std.debug.print("Gamma_tot = {d}\n", .{gamma.value});
    std.debug.print("dGamma_tot = {d}\n", .{deriv});
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalFWHM = @import("./NormalFWHM.zig");
const LorentzFWHM = @import("./LorentzFWHM.zig");
