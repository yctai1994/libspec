eta: *PseudoVoigtEta = undefined,
gamma: *PseudoVoigtGamma = undefined,
normal: *PseudoVoigtNormal = undefined,
lorentz: *PseudoVoigtLorentz = undefined,

const Self = @This();

pub fn init(allocator: mem.Allocator) !*Self {
    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.eta = try PseudoVoigtEta.init(allocator);
    errdefer self.eta.deinit(allocator);

    self.gamma = self.eta.gamma;

    self.normal = try allocator.create(PseudoVoigtNormal);
    errdefer allocator.destroy(self.normal);

    self.lorentz = try allocator.create(PseudoVoigtLorentz);

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.eta.deinit(allocator);
    allocator.destroy(self.normal);
    allocator.destroy(self.lorentz);
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;
    const this: *Self = try Self.init(page);
    defer this.deinit(page);
}

pub fn forward(self: *Self, mu: f64, sigma: f64, gamma: f64) void {
    self.eta.forward(sigma, gamma);
    self.normal.forward(mu, self.gamma);
    self.lorentz.forward(mu, self.gamma);
    return;
}

pub fn writeAll(self: *Self, parr: []f64, xarr: []f64) void {
    const eta: f64 = self.eta.value;
    const tmp: f64 = 1.0 - eta;

    for (parr, xarr) |*p_i, x_i| {
        p_i.* = eta * self.lorentz.density(x_i) + tmp * self.normal.density(x_i);
    }
    return;
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtEta = @import("./PseudoVoigtEta.zig");
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
const PseudoVoigtNormal = @import("./PseudoVoigtNormal.zig");
const PseudoVoigtLorentz = @import("./PseudoVoigtLorentz.zig");
