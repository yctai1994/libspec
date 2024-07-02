// Tape: [..., dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined,
deriv: f64 = undefined, // dΓtot/dΓG
deriv_in: *f64 = undefined, // dy/dΓtot
deriv_out: *f64 = undefined, // dy/dΓG

scale: *NormalScale = undefined,

const Self = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 3) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try NormalScale.init(allocator, &tape[1]);

    self.deriv_in = &tape[0];
    self.deriv_out = &tape[1];

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, scale: f64) void {
    const temp: comptime_float = comptime 2.0 * @sqrt(2.0 * @log(2.0));
    self.scale.forward(scale);
    self.scale.deriv = temp; // dΓG/dσ
    self.value = temp * scale;
    return;
}

pub fn backward(self: *Self, final_deriv_out: *f64) void {
    // dy/dΓG = (dΓtot/dΓG) × (dy/dΓtot)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    self.scale.backward(final_deriv_out); // final_deriv_out := &(dy/dσ)
    return;
}

const std = @import("std");
const mem = std.mem;

const NormalScale = @import("./NormalScale.zig");
