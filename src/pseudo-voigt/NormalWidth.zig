//! Normal Distribution Full Width at Half Maximum (FWHM)
value: f64, // FN
deriv: f64, // dFᵥ/dFN
deriv_in: *f64, // dy/dFᵥ
deriv_out: *f64, // dy/dFN

scale: *NormalScale,

const Self: type = @This(); // hosted by PseudoVoigtWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 5 * n;
    if (tape.len != m + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try NormalScale.init(allocator, tape, n);

    self.deriv_in = &tape[m + 3]; // dy/dFᵥ
    self.deriv_out = &tape[m + 4]; // dy/dFN

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
}

pub fn forward(self: *Self, scale: f64) void {
    const temp: comptime_float = comptime 2.0 * @sqrt(2.0 * @log(2.0));
    self.scale.forward(scale);
    self.scale.deriv = temp; // dFN/dσ
    self.value = temp * scale;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // dy/dFN = (dFᵥ/dFN) × (dy/dFᵥ)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    return self.scale.backward(final_deriv_out);
}

test "NormalWidth: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 5 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self: *Self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(test_sigma);
    self.deriv = 1.0; // only need for unit-testing
    self.backward(dest);

    try testing.expectEqual(0x1.473028646a507p2, self.value);
    try testing.expectEqual(0x1.2d6abe44afc43p1, dest[1]);
}

const test_n: comptime_int = 1;
const test_sigma: comptime_float = 2.171;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalScale = @import("./NormalScale.zig");
