//! Normal Distribution Full Width at Half Maximum (FWHM)
value: f64 = undefined, // FN
deriv: f64 = undefined, // dFᵥ/dFN
deriv_in: *f64 = undefined, // dy/dFᵥ
deriv_out: *f64 = undefined, // dy/dFN

scale: *NormalScale,

const Self: type = @This(); // hosted by PseudoVoigtWidth

pub fn init(allocator: mem.Allocator, tape: []f64) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try NormalScale.init(allocator, tape);

    // self.deriv_in = &tape[TBD]; // dy/dFᵥ
    // self.deriv_out = &tape[TBD]; // dy/dFN

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
    return;
}

fn forward(self: *Self, scale: f64) void {
    const temp: comptime_float = comptime 2.0 * @sqrt(2.0 * @log(2.0));
    self.scale.forward(scale);
    self.scale.deriv = temp; // dFN/dσ
    self.value = temp * scale;
    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

    // dy/dFN = (dFᵥ/dFN) × (dy/dFᵥ)
    self.deriv_out.* = self.deriv * self.deriv_in.*;

    self.scale.backward(final_deriv_out);

    return;
}

test "init" {
    const page = testing.allocator;
    const self = try Self.init(page, &.{});
    defer self.deinit(page);
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalScale = @import("./NormalScale.zig");
