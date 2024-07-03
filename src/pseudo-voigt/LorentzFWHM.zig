// Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined,
deriv: [2]f64 = undefined, // [ dη/dΓL, dΓtot/dΓL ]
deriv_in: []f64 = undefined, // [ dy/dη, dy/dΓtot ]
deriv_out: *f64 = undefined, // dy/dΓL

scale: *LorentzScale = undefined,

const Self = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try LorentzScale.init(allocator, tape);

    self.deriv = .{ 0.0, 1.0 };
    self.deriv_in = tape[5..7];
    self.deriv_out = &tape[8];

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
    // (dy/dΓL) = [ dη/dΓL, dΓtot/dΓL ]ᵀ ⋅ [ dy/dη, dy/dΓtot ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| {
        temp += deriv * deriv_in;
    }

    self.deriv_out.* = temp;
    self.scale.backward(final_deriv_out); // final_deriv_out := &(dy/dγ)
    return;
}

const std = @import("std");
const mem = std.mem;

const LorentzScale = @import("./LorentzScale.zig");
