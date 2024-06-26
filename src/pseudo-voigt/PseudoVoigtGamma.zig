value: f64 = undefined, // Γ
deriv: [2]f64 = undefined, // dΓ/dσ, dΓ/dγ

normal: *NormalFWHM = undefined,
lorentz: *LorentzFWHM = undefined,

backproped: bool = undefined,

const G0: comptime_float = 1.0;
const G1: comptime_float = 0.07842;
const G2: comptime_float = 4.47163;
const G3: comptime_float = 2.42843;
const G4: comptime_float = 2.69269;
const G5: comptime_float = 1.0;

const Self: type = @This();

pub fn init(allocator: mem.Allocator) !*Self {
    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.normal = try NormalFWHM.init(allocator);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try LorentzFWHM.init(allocator);

    self.backproped = false;

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.normal.deinit(allocator);
    self.lorentz.deinit(allocator);
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;
    const this: *Self = try Self.init(page);
    defer this.deinit(page);
}

pub fn forward(self: *Self, sigma: f64, gamma: f64) void {
    self.normal.forward(sigma);
    self.lorentz.forward(gamma);

    const GammaG: f64 = self.normal.value;
    const GammaL: f64 = self.lorentz.value;

    var cGtot: [6]f64 = comptime .{ G0, G1, G2, G3, G4, G5 };
    var cdGds: [5]f64 = comptime .{ G1 * 1.0, G2 * 2.0, G3 * 3.0, G4 * 4.0, G5 * 5.0 };
    var cdGdg: [5]f64 = comptime .{ G0 * 5.0, G1 * 4.0, G2 * 3.0, G3 * 2.0, G4 * 1.0 };

    var temp: f64 = GammaG;
    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |*c_Gtot, *c_dGds, *c_dGdg| {
        c_Gtot.* *= temp;
        c_dGds.* *= temp;
        c_dGdg.* *= temp;
        temp *= GammaG;
    }
    cGtot[5] *= temp;

    var Gtot: f64 = cGtot[0];
    var dGds: f64 = cdGds[0];
    var dGdg: f64 = cdGdg[0];

    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |c_Gtot, c_dGds, c_dGdg| {
        Gtot = Gtot * GammaL + c_Gtot;
        dGds = dGds * GammaL + c_dGds;
        dGdg = dGdg * GammaL + c_dGdg;
    }
    Gtot = Gtot * GammaL + cGtot[5];

    temp = @log(Gtot);
    Gtot = @exp(0.2 * temp);
    temp = @exp(-0.8 * temp) * 0.2;

    self.value = Gtot;
    self.deriv[0] = temp * dGds;
    self.deriv[1] = temp * dGdg;

    self.backproped = false;

    return;
}

pub fn backward(self: *Self) void {
    if (!self.backproped) {
        self.deriv[0] *= self.normal.backward(); // dΓ/dσ
        self.deriv[1] *= self.lorentz.backward(); // dΓ/dγ
        self.backproped = true;
    }
    return;
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalFWHM = @import("./NormalFWHM.zig");
const LorentzFWHM = @import("./LorentzFWHM.zig");
