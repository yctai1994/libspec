//! PseudoLorentzScale.zig
//! Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // γV
deriv: f64 = undefined, // dPL/dγV
deriv_in: *f64 = undefined, // dy/dPL
deriv_out: *f64 = undefined, // dy/dγV

gamma: *PseudoVoigtGamma = undefined,

const Self: type = @This();

pub fn init(allocator: mem.Allocator, tape: []f64, gamma: *PseudoVoigtGamma) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);

    self.deriv = 1.0; // should be removed later
    self.deriv_in = &tape[2];
    self.deriv_out = &tape[4];

    self.gamma = gamma;

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;

    var tape: []f64 = try page.alloc(f64, 9);
    defer page.free(tape);
    _ = &tape;

    const gamma: *PseudoVoigtGamma = try PseudoVoigtGamma.init(page, tape);
    defer gamma.deinit(page);

    const scale: *Self = try Self.init(page, tape, gamma);
    defer scale.deinit(page);
}

pub fn forward(self: *Self) void {
    // Gamma should be already forwarded.
    self.gamma.deriv[1] = 0.5; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
    self.value = 0.5 * self.gamma.value;

    return;
}

pub fn backward(self: *Self) void {
    // (dy/dγV) = (dPL/dγV) × (dy/dPL)
    self.deriv_out.* = self.deriv * self.deriv_in.*;

    return;
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
