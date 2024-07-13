//! Pseudo-Voigt Normal Scale
value: f64, // σᵥ
deriv: []f64, // [ dPN₁/dσᵥ, dPN₂/dσᵥ, … ]
deriv_in: []f64, // [ dy/dPN₁, dy/dPN₂, … ]
deriv_out: *f64, // dy/dσᵥ

width: *PseudoVoigtWidth, // hosted by PseudoVoigtLogL

const Self: type = @This(); // hosted by PseudoVoigtNormal

pub fn init(allocator: mem.Allocator, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    if (tape.len != 5 * n + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);
    self.width = width;

    const m: usize = 2 * n;
    self.deriv_in = tape[m .. m + n]; // [ dy/dPN₁, dy/dPN₂, … ]
    self.deriv_out = &tape[m + m + n]; // dy/dσᵥ

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
}

pub fn forward(self: *Self) void {
    // PseudoVoigtWidth should be already forwarded.
    const temp: comptime_float = comptime 0.5 / @sqrt(2.0 * @log(2.0));
    self.width.deriv[0] = temp; // [ dσᵥ/dFᵥ, dγᵥ/dFᵥ, dη/dFᵥ ]
    self.value = temp * self.width.value;
}

pub fn backward(self: *Self) void {
    // (dy/dσᵥ) = [ dPN₁/dσᵥ, dPN₂/dσᵥ, … ]ᵀ⋅[ dy/dPN₁, dy/dPN₂, … ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |d, din| temp += d * din;
    self.deriv_out.* = temp;
}

test "PseudoNormalScale: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 5 * test_n + 6);
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

    try testing.expectApproxEqRel(0x1.67e08da02e0e6p+1, self.value, 2e-16);
    try testing.expectApproxEqRel(0x1.eedb2e2a66fbbp-1, dest[1], 4e-16);
    try testing.expectApproxEqRel(0x1.17d4097e602d2p-1, dest[2], 7e-16);
}

const test_n: comptime_int = 1;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
