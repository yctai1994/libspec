//! Normal Distribution Full Width at Half Maximum (FWHM)
value: f64 = undefined, // FN
deriv: f64 = undefined, // dFᵥ/dFN
deriv_in: *f64 = undefined, // dy/dFᵥ
deriv_out: *f64 = undefined, // dy/dFN

scale: *NormalScale,

const Self: type = @This(); // hosted by PseudoVoigtWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try NormalScale.init(allocator, tape, n);

    self.deriv_in = &tape[4 * n + 3]; // dy/dFᵥ
    self.deriv_out = &tape[4 * n + 4]; // dy/dFN

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

test "NormalWidth: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(test_sigma);
    self.deriv = 1.0; // only need for unit-testing
    self.backward(dest);

    try testing.expectEqual(0x1.473028646a507p2, self.value);
    try testing.expectEqual(0x1.2d6abe44afc43p1, dest[1]);
}

const test_n: comptime_int = 0;
const test_sigma: comptime_float = 2.171;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalScale = @import("./NormalScale.zig");
