//! Pseudo-Voigt Ratio of Lorentz Distribution
value: f64, // η
deriv: []f64, // [ dPv₁/dη, dPv₂/dη, … ]
deriv_in: []f64, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: *f64, // dy/dη

width: *PseudoVoigtWidth, // hosted by PseudoVoigtLogL
lorentz: *LorentzWidth, // hosted by PseudoVoigtWidth

const Self: type = @This(); // hosted by PseudoVoigt

const RATIO_FAC0: comptime_float = 0.0; // η₀
const RATIO_FAC1: comptime_float = 1.36603; // η₁
const RATIO_FAC2: comptime_float = -0.47719; // η₂
const RATIO_FAC3: comptime_float = 0.11116; // η₃

pub fn init(allocator: mem.Allocator, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    const m: usize = 5 * n;
    if (tape.len != m + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);

    self.width = width;
    self.lorentz = width.lorentz;

    self.deriv_in = tape[n .. n * 2]; // [ dPv₁/dη, dPv₂/dη, … ]
    self.deriv_out = &tape[m + 2]; // dy/dη

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
}

pub fn forward(self: *Self) void {
    // PseudoVoigtWidth should be already forwarded.
    const alpha: f64 = self.lorentz.value / self.width.value; // α = FL/Fᵥ

    var eta: f64 = RATIO_FAC3; // η = η₀ + η₁α + η₂α² + η₃α³
    var beta: f64 = 0.0; // β = dη/dα = η₁ + 2η₂α + 3η₃α²

    inline for (.{ RATIO_FAC2, RATIO_FAC1, RATIO_FAC0 }) |coeff| {
        beta = beta * alpha + eta;
        eta = eta * alpha + coeff;
    }

    self.value = eta;

    beta /= self.width.value;

    self.width.deriv[2] = -alpha * beta; // [ dσᵥ/dFᵥ, dγᵥ/dFᵥ, dη/dFᵥ ]
    self.lorentz.deriv[0] = beta; // [ dη/dFL, dFᵥ/dFL ]
}

pub fn backward(self: *Self) void {
    // (dy/dη) = [ dPv₁/dη, dPv₂/dη, … ]ᵀ⋅[ dy/dPv₁, dy/dPv₂, … ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |d, din| temp += d * din;
    self.deriv_out.* = temp;
}

test "PseudoVoigtRatio: forward & backward" {
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

    @memset(self.deriv, 1.0); // only need for unit-testing
    self.forward();

    self.backward();
    width.backward(dest);

    try testing.expectApproxEqRel(0x1.e279811e1afa5p-2, self.value, 2e-16);
    try testing.expectApproxEqRel(-0x1.2118a9da8538ep-3, dest[1], 1e-15);
    try testing.expectApproxEqRel(0x1.e0f0ed0ebe3dfp-3, dest[2], 4e-16);
}

const test_n: comptime_int = 1;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const LorentzWidth = @import("./LorentzWidth.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
