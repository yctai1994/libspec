// Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // σ
deriv: f64 = undefined, // dΓG/dσ
deriv_in: *f64 = undefined, // dy/dΓG

const Self: type = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    self.deriv = 1.0; // should be removed later
    self.deriv_in = &tape[7];
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
    // dy/dσ = (dΓG/dσ) × (dy/dΓG), deriv_out := &(dy/dσ)
    deriv_out.* = self.deriv * self.deriv_in.*;
    return;
}

const std = @import("std");
const mem = std.mem;

const NormalFWHM = @import("./NormalFWHM.zig");
