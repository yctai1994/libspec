//! Pseudo-Voigt Normal
value: []f64 = undefined, // [ N(x̄₁, σᵥ), N(x̄₂, σᵥ), … ]
deriv: []f64 = undefined, // [ dPv₁/dPN₁, dPv₂/dPN₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPN₁, dy/dPN₂, … ]

cdata: *CenteredData, // hosted by PseudoVoigtLogL
scale: *PseudoNormalScale,

const Self: type = @This(); // hosted by PseudoVoigt

// Called by PseudoVoigt
fn init(allocator: mem.Allocator, cdata: *CenteredData, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    const m: usize = 2 * n;
    if (tape.len != (m <<| 1) + n + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    self.scale = try PseudoNormalScale.init(allocator, width, tape, n);
    self.cdata = cdata;

    self.deriv_in = tape[n..m]; // [ dy/dPv₁, dy/dPv₂, … ]
    self.deriv_out = tape[m .. m + n]; // [ dy/dPN₁, dy/dPN₂, … ]

    return self;
}

// Called by PseudoVoigt
fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
    return;
}

// Called by PseudoVoigt
fn forward(self: *Self) void {
    self.scale.forward();

    for (self.value, self.cdata.value) |*prob, centered_x| {
        prob.* = density(centered_x, self.scale.value);
    }

    const n: usize = self.value.len;
    const inv_sigma: f64 = 1.0 / self.scale.value;

    // arg1ᵢ = x̄ᵢ/σᵥ, arg2ᵢ = PNᵢ/σᵥ
    for (self.deriv, self.deriv_out, self.cdata.value, self.value) |*arg1, *arg2, centered_x, prob| {
        arg1.* = inv_sigma * centered_x;
        arg2.* = inv_sigma * prob;
    }

    for (self.deriv, self.deriv_out, self.scale.deriv, self.cdata.deriv[0..n]) |arg1, arg2, *dsigma, *dcentered_x| {
        dsigma.* = arg1 * arg2; // dPNᵢ/dσᵥ ← (arg1ᵢ)⋅(arg2ᵢ)
        dcentered_x.* = -dsigma.*; // dPNᵢ/dx̄ᵢ = -(arg1ᵢ)⋅(arg2ᵢ)
    }

    for (self.deriv, self.deriv_out, self.scale.deriv) |arg1, arg2, *dsigma| {
        // dPNᵢ/dσᵥ = (arg1ᵢ)²⋅(arg2ᵢ) - arg2ᵢ
        dsigma.* = dsigma.* * arg1 - arg2;
    }

    return;
}

// Called by PseudoVoigt
fn backward(self: *Self) void {
    // [ dy/dPN₁, dy/dPN₂, … ] = [ dPv₁/dPN₁, dPv₂/dPN₂, … ]ᵀ⋅[ dy/dPv₁, dy/dPv₂, … ]
    for (self.deriv_out, self.deriv, self.deriv_in) |*dout, d, din| dout.* = d * din;
    return self.scale.backward();
}

inline fn density(centered_x: f64, scale: f64) f64 {
    const temp: comptime_float = comptime 1.0 / @sqrt(2.0 * math.pi);
    return temp * @exp(-0.5 * pow2(centered_x / scale)) / scale;
}

fn pow2(x: f64) f64 {
    return x * x;
}

test "PseudoVoigtNormal: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 5 * test_n + 6);
    defer page.free(tape);

    @memset(tape, 1.0);

    const cdata: *CenteredData = try CenteredData.init(page, tape, test_n);
    defer cdata.deinit(page);

    const width: *PseudoVoigtWidth = try PseudoVoigtWidth.init(page, tape, test_n);
    defer width.deinit(page);

    const self: *Self = try Self.init(page, cdata, width, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    var xvec: [1]f64 = .{test_x};

    @memset(&width.deriv, 0.0); // only need for unit-testing

    cdata.forward(&xvec, test_mode);
    width.forward(test_sigma, test_gamma);
    self.forward();

    @memset(self.deriv, 1.0); // only need for unit-testing

    self.backward();
    cdata.backward(dest);
    width.backward(dest);

    try testing.expectApproxEqRel(0x1.208b65b17ea48p-3, self.value[0], 2e-16);
    try testing.expectApproxEqRel(0x1.874ee84c784c8p-8, dest[0], 6e-16);
    try testing.expectApproxEqRel(-0x1.8722c64450ac2p-5, dest[1], 9e-16);
    try testing.expectApproxEqRel(-0x1.ba5aafe3f2d82p-6, dest[2], 2e-15);
}

const test_n: comptime_int = 1;
const test_x: comptime_float = 1.213;
const test_mode: comptime_float = 0.878;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
const PseudoNormalScale = @import("./PseudoNormalScale.zig");
