//! Pseudo-Voigt Full Width at Half Maximum (FWHM)
value: f64, // Fᵥ
deriv: [3]f64, // [ dσᵥ/dFᵥ, dγᵥ/dFᵥ, dη/dFᵥ ]
deriv_in: []f64, // [ dy/dσᵥ, dy/dγᵥ, dy/dη ]
deriv_out: *f64, // dy/dFᵥ

normal: *NormalWidth,
lorentz: *LorentzWidth,

const Self: type = @This(); // hosted by PseudoVoigtLogL

const WIDTH_FAC0: comptime_float = 1.0;
const WIDTH_FAC1: comptime_float = 0.07842;
const WIDTH_FAC2: comptime_float = 4.47163;
const WIDTH_FAC3: comptime_float = 2.42843;
const WIDTH_FAC4: comptime_float = 2.69269;
const WIDTH_FAC5: comptime_float = 1.0;

fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.normal = try NormalWidth.init(allocator, tape, n);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try LorentzWidth.init(allocator, tape, n);

    const m: usize = 4 * n;
    self.deriv_in = tape[m .. m + 3]; // [ dy/dσᵥ, dy/dγᵥ, dy/dη ]
    self.deriv_out = &tape[m + 3]; // dy/dFᵥ

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.lorentz.deinit(allocator);
    self.normal.deinit(allocator);
    allocator.destroy(self);
}

fn forward(self: *Self, sigma: f64, gamma: f64) void {
    self.normal.forward(sigma);
    self.lorentz.forward(gamma);

    const normal_width: f64 = self.normal.value;
    const lorentz_width: f64 = self.lorentz.value;

    var width_buffer: [6]f64 = comptime .{
        WIDTH_FAC0,
        WIDTH_FAC1,
        WIDTH_FAC2,
        WIDTH_FAC3,
        WIDTH_FAC4,
        WIDTH_FAC5,
    };

    var normal_buffer: [5]f64 = comptime .{
        WIDTH_FAC1 * 1.0,
        WIDTH_FAC2 * 2.0,
        WIDTH_FAC3 * 3.0,
        WIDTH_FAC4 * 4.0,
        WIDTH_FAC5 * 5.0,
    };

    var lorentz_buffer: [5]f64 = comptime .{
        WIDTH_FAC0 * 5.0,
        WIDTH_FAC1 * 4.0,
        WIDTH_FAC2 * 3.0,
        WIDTH_FAC3 * 2.0,
        WIDTH_FAC4 * 1.0,
    };

    var temp: f64 = normal_width;

    for (width_buffer[1..5], normal_buffer[1..], lorentz_buffer[1..]) |*width_coeff, *normal_coeff, *lorentz_coeff| {
        width_coeff.* *= temp;
        normal_coeff.* *= temp;
        lorentz_coeff.* *= temp;
        temp *= normal_width;
    }
    width_buffer[5] *= temp;

    var pv_width: f64 = width_buffer[0];
    var normal_partial: f64 = normal_buffer[0];
    var lorentz_partial: f64 = lorentz_buffer[0];

    for (width_buffer[1..5], normal_buffer[1..], lorentz_buffer[1..]) |width_coeff, normal_coeff, lorentz_coeff| {
        pv_width = pv_width * lorentz_width + width_coeff;
        normal_partial = normal_partial * lorentz_width + normal_coeff;
        lorentz_partial = lorentz_partial * lorentz_width + lorentz_coeff;
    }
    pv_width = pv_width * lorentz_width + width_buffer[5];

    temp = @log(pv_width);
    self.value = @exp(0.2 * temp);

    temp = @exp(-0.8 * temp) * 0.2;

    self.normal.deriv = temp * normal_partial; // dFᵥ/dFN
    self.lorentz.deriv[1] = temp * lorentz_partial; // [ dη/dFL, dFᵥ/dFL ]
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    // (dy/dFᵥ) = [ dσᵥ/dFᵥ, dγᵥ/dFᵥ, dη/dFᵥ ]ᵀ⋅[ dy/dσᵥ, dy/dγᵥ, dy/dη ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| {
        temp += deriv * deriv_in;
    }
    self.deriv_out.* = temp;

    self.normal.backward(final_deriv_out);
    self.lorentz.backward(final_deriv_out);
}

test "PseudoVoigtWidth: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 4 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(test_sigma, test_gamma);
    @memset(&self.deriv, 0x1.5555555555555p-2); // only need for unit-testing
    self.backward(dest);

    try testing.expectApproxEqRel(0x1.a7b914f93252cp2, self.value, 1e-15);
    try testing.expectApproxEqRel(0x1.235305ea3571bp1, dest[1], 1e-15);
    try testing.expectApproxEqRel(0x1.4978fceff8e7ap0, dest[2], 1e-15);
}

const test_n: comptime_int = 0;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const NormalWidth = @import("./NormalWidth.zig");
const LorentzWidth = @import("./LorentzWidth.zig");
