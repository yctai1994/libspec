//! Pseudo-Voigt Lorentz Scale
value: f64 = undefined, // γᵥ
deriv: []f64 = undefined, // [ dPL₁/dγᵥ, dPL₂/dγᵥ, … ]
deriv_in: []f64 = undefined, // [ dy/dPL₁, dy/dPL₂, … ]
deriv_out: *f64 = undefined, // dy/dγᵥ

width: *PseudoVoigtWidth, // hosted by PseudoVoigtLogL

const Self: type = @This(); // hosted by PseudoVoigtLorentz

pub fn init(allocator: mem.Allocator, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);
    self.width = width;

    const m: usize = 3 * n;
    self.deriv_in = tape[m .. m + n]; // [ dy/dPL₁, dy/dPL₂, … ]
    self.deriv_out = &tape[m + n + 1]; // dy/dγᵥ

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
}

pub fn forward(self: *Self) void {
    // PseudoVoigtWidth should be already forwarded.
    self.width.deriv[1] = 0.5; // [ dσᵥ/dFᵥ, dγᵥ/dFᵥ, dη/dFᵥ ]
    self.value = 0.5 * self.width.value;
}

pub fn backward(self: *Self) void {
    // (dy/dγᵥ) = [ dPL₁/dγᵥ, dPL₂/dγᵥ, … ]ᵀ⋅[ dy/dPL₁, dy/dPL₂, … ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |d, din| temp += d * din;
    self.deriv_out.* = temp;
}

test "PseudoLorentzScale: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const width: *PseudoVoigtWidth = try PseudoVoigtWidth.init(page, tape, test_n);
    defer width.deinit(page);

    const self: *Self = try Self.init(page, width, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    @memset(&width.deriv, 0.0); // only need for unit-testing

    width.forward(test_sigma, test_gamma);
    self.forward();

    @memset(self.deriv, 1.0); // only need for unit-testing

    self.backward();
    width.backward(dest);

    try testing.expectApproxEqRel(0x1.a7b914f93252cp+1, self.value, 3e-16);
    try testing.expectApproxEqRel(0x1.235305ea3571bp+0, dest[1], 4e-16);
    try testing.expectApproxEqRel(0x1.4978fceff8e7ap-1, dest[2], 7e-16);
}

const test_n: comptime_int = 1;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
