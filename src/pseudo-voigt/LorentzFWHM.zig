// Tape: [..., dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined,
deriv: f64 = undefined, // dΓtot/dΓL
deriv_in: *f64 = undefined, // dy/dΓtot
deriv_out: *f64 = undefined, // dy/dΓL

scale: *LorentzScale = undefined,

const Self = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 3) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try LorentzScale.init(allocator, &tape[2]);

    self.deriv_in = &tape[0];
    self.deriv_out = &tape[2];

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, scale: f64) void {
    self.scale.forward(scale);
    self.scale.deriv = 2.0; // dΓL/dγ
    self.value = 2.0 * scale;
    return;
}

pub fn backward(self: *Self, final_deriv_out: *f64) void {
    // dy/dΓL = (dΓtot/dΓL) × (dy/dΓtot)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    self.scale.backward(final_deriv_out); // final_deriv_out := &(dy/dγ)
    return;
}

const std = @import("std");
const mem = std.mem;

const LorentzScale = @import("./LorentzScale.zig");
