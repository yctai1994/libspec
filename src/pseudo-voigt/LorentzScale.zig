//! Lorentz Distribution Scale
value: f64, // γ
deriv: f64, // dFL/dγ
deriv_in: *f64, // dy/dFL

const Self: type = @This(); // hosted by LorentzWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 5 * n;
    if (tape.len != m + 6) unreachable;

    const self = try allocator.create(Self);
    self.deriv_in = &tape[m + 5]; // dy/dFL
    return self;
}

pub inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
}

pub inline fn forward(self: *Self, scale: f64) void {
    self.value = scale;
}

pub fn backward(self: *Self, deriv_out: []f64) void {
    if (deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]
    deriv_out[2] = self.deriv * self.deriv_in.*; // dy/dγ = (dFL/dγ) × (dy/dFL)
}

test "init" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 5 * test_n + 6);
    defer page.free(tape);

    const self: *Self = try Self.init(page, tape, test_n);
    defer self.deinit(page);
}

const test_n: comptime_int = 1;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
