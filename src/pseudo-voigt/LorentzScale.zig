// Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // γ
deriv: f64 = undefined, // dΓL/dγ
deriv_in: *f64 = undefined, // dy/dΓL

const Self: type = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    self.deriv = 1.0; // should be removed later
    self.deriv_in = &tape[8];
    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, scale: f64) void {
    self.value = scale;
    return;
}

pub fn backward(self: *Self, deriv_out: *f64) void {
    // dy/dγ = (dΓL/dγ) × (dy/dΓL), deriv_out := &(dy/dγ)
    deriv_out.* = self.deriv * self.deriv_in.*;
    return;
}

const std = @import("std");
const mem = std.mem;

const LorentzFWHM = @import("./LorentzFWHM.zig");
