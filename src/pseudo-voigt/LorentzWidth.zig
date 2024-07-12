//! Lorentz Distribution Full Width at Half Maximum (FWHM)
value: f64 = undefined, // FL
deriv: f64 = undefined, // dFᵥ/dFL
deriv_in: *f64 = undefined, // dy/dFᵥ
deriv_out: *f64 = undefined, // dy/dFL

scale: *LorentzScale,

const Self: type = @This(); // hosted by PseudoVoigtWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try LorentzScale.init(allocator, tape, n);

    self.deriv_in = &tape[4 * n + 3]; // dy/dFᵥ
    self.deriv_out = &tape[4 * n + 5]; // dy/dFL

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
    return;
}

fn forward(self: *Self, scale: f64) void {
    self.scale.forward(scale);
    self.scale.deriv = 2.0; // dFL/dγ
    self.value = 2.0 * scale;
    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    // dy/dFL = (dFᵥ/dFL) × (dy/dFᵥ)
    self.deriv_out.* = self.deriv * self.deriv_in.*;
    return self.scale.backward(final_deriv_out);
}

test "LorentzWidth: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(test_gamma);
    self.deriv = 1.0; // only need for unit-testing
    self.backward(dest);

    try testing.expectEqual(0x1.4e147ae147ae1p1, self.value);
    try testing.expectEqual(0x1.0000000000000p1, dest[2]);
}

const test_n: comptime_int = 0;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const LorentzScale = @import("./LorentzScale.zig");
