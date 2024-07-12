//! Lorentz Distribution Full Width at Half Maximum (FWHM)
value: f64, // FL
deriv: [2]f64, // [ dη/dFL, dFᵥ/dFL ]
deriv_in: []f64, // [ dy/dη, dy/dFᵥ ]
deriv_out: *f64, // dy/dFL

scale: *LorentzScale,

const Self: type = @This(); // hosted by PseudoVoigtWidth

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.scale = try LorentzScale.init(allocator, tape, n);

    const m: usize = 4 * n;
    self.deriv_in = tape[m + 2 .. m + 4]; // [ dy/dη, dy/dFᵥ ]
    self.deriv_out = &tape[m + 5]; // dy/dFL

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.destroy(self);
}

pub fn forward(self: *Self, scale: f64) void {
    self.scale.forward(scale);
    self.scale.deriv = 2.0; // dFL/dγ
    self.value = 2.0 * scale;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // (dy/dFL) = [ dη/dFL, dFᵥ/dFL ]ᵀ⋅[ dy/dη, dy/dFᵥ ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| temp += deriv * deriv_in;
    self.deriv_out.* = temp;
    return self.scale.backward(final_deriv_out);
}

test "LorentzWidth: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self: *Self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(test_gamma);
    self.deriv[0] = 0.0; // dη/dFL, only need for unit-testing
    self.deriv[1] = 1.0; // dFᵥ/dFL, only need for unit-testing
    self.backward(dest);

    try testing.expectEqual(0x1.4e147ae147ae1p1, self.value);
    try testing.expectEqual(0x1.0000000000000p1, dest[2]);
}

const test_n: comptime_int = 1;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const LorentzScale = @import("./LorentzScale.zig");
