//! Normal Distribution Scale
value: f64 = undefined, // σ
deriv: f64 = undefined, // dFN/dσ
deriv_in: *f64 = undefined, // dy/dFN

const Self: type = @This(); // hosted by NormalWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);

    self.deriv_in = &tape[4 * n + 4]; // dy/dFN

    return self;
}

pub inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

pub inline fn forward(self: *Self, scale: f64) void {
    self.value = scale;
    return;
}

pub fn backward(self: *Self, deriv_out: []f64) void {
    if (deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]
    deriv_out[1] = self.deriv * self.deriv_in.*; // dy/dσ = (dFN/dσ) × (dy/dFN)
    return;
}

test "init" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    const self = try Self.init(page, tape, test_n);
    defer self.deinit(page);
}

const test_n: comptime_int = 0;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
