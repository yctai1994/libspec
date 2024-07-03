//! PseudoNormalScale.zig
//! Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // σV
deriv: f64 = undefined, // dPG/dσV
deriv_in: *f64 = undefined, // dy/dPG
deriv_out: *f64 = undefined, // dy/dσV

gamma: *PseudoVoigtGamma = undefined,

const Self: type = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    self.deriv = 1.0; // should be removed later
    self.deriv_in = &tape[1];
    self.deriv_in = &tape[3];
    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, scale: f64) void {
    const temp: comptime_float = comptime 2.0 * @sqrt(2.0 * @log(2.0));
    self.gamma.deriv[0] = comptime 1.0 / temp; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
    self.value = scale;
    return;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // final_deriv_out := [dy/dσ, dy/dγ]
    if (final_deriv_out.len != 2) unreachable;

    // (dy/dσV) = (dPG/dσV) × (dy/dPG)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    return;
}

const std = @import("std");
const mem = std.mem;

const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
