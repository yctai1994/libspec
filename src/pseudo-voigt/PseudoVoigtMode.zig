//! PseudoVoigtMode.zig
//! Tape: [dy/dPpV, dy/dPG, dy/dPL, dy/dσV, dy/dγV, dy/dη, dy/dΓtot, dy/dΓG, dy/dΓL]
value: f64 = undefined, // μ
deriv: [2]f64 = undefined, // [ dPG/dμ, dPL/dμ ]
deriv_in: []f64 = undefined, // [ dy/dPG, dy/dPL ]

const Self: type = @This();

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    if (tape.len != 9) unreachable;

    var self: *Self = try allocator.create(Self);

    self.deriv = .{ 0.0, 1.0 }; // should be removed later
    self.deriv_in = tape[1..3];

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, mode: f64) void {
    self.value = mode;
    return;
}

pub fn backward(self: *Self, deriv_out: *f64) void {
    // (dy/dμ) = [ dPG/dμ, dPL/dμ ]ᵀ ⋅ [ dy/dPG, dy/dPL ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| {
        temp += deriv * deriv_in;
    }

    deriv_out.* = temp;
    return;
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
