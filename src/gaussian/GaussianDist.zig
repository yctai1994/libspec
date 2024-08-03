//! Gaussian Distribution
value: []f64, // [ N(x̄₁, σ), N(x̄₂, σ), … ]
deriv: []f64, // [ d(-logL)/d(logPN₁), d(-logL)/d(logPN₂), … ]
deriv_out: []f64, // [ dy/dPN₁, dy/dPN₂, … ]

cdata: *CenteredData,
scale: *GaussianScale,

const Self: type = @This(); // hosted by GaussianLogL

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 2 * n;
    if (tape.len != m + 1) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    self.cdata = try CenteredData.init(allocator, tape, n);
    errdefer self.cdata.deinit(allocator);

    self.scale = try GaussianScale.init(allocator, tape, n);

    self.deriv_out = tape[0..n]; // [ dy/dPN₁, dy/dPN₂, … ]

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    self.cdata.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
    return;
}

pub fn forward(self: *Self, xvec: []f64, mode: f64, scale: f64) void {
    self.cdata.forward(xvec, mode);
    self.scale.forward(scale);

    for (self.value, self.cdata.value) |*prob, centered_x| prob.* = density(centered_x, scale);

    const inv_scale: f64 = 1.0 / scale;

    var arg1: f64 = undefined;
    var arg2: f64 = undefined;

    for (self.scale.deriv, self.cdata.deriv, self.cdata.value, self.value) |*dsigma, *dcentered_x, centered_x, prob| {
        arg1 = inv_scale * centered_x; // arg1ᵢ = x̄ᵢ/σ
        arg2 = inv_scale * prob; // arg2ᵢ = PNᵢ/σ

        dsigma.* = arg1 * arg2; // dPNᵢ/dσ ← (arg1ᵢ)⋅(arg2ᵢ)
        dcentered_x.* = -dsigma.*; // dPNᵢ/dx̄ᵢ = -(arg1ᵢ)⋅(arg2ᵢ)

        dsigma.* = dsigma.* * arg1 - arg2; // dPNᵢ/dσ = (arg1ᵢ)²⋅(arg2ᵢ) - arg2ᵢ
    }
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // [ dy/dPN₁, dy/dPN₂, … ] = [ d(-logL)/d(logPN₁), d(-logL)/d(logPN₂), … ]ᵀ⋅[ dy/d(-logL), dy/d(-logL), … ]
    for (self.deriv_out, self.deriv, self.value) |*dout, minus_w, p| dout.* = minus_w / p;
    self.cdata.backward(final_deriv_out);
    self.scale.backward(final_deriv_out);
}

inline fn density(centered_x: f64, scale: f64) f64 {
    const temp: comptime_float = comptime 1.0 / @sqrt(2.0 * math.pi);
    return temp * @exp(-0.5 * pow2(centered_x / scale)) / scale;
}

fn pow2(x: f64) f64 {
    return x * x;
}

test "GaussianDist: forward & backward" {
    const page = testing.allocator;

    const tape: []f64 = try page.alloc(f64, 2 * test_n + 1);
    defer page.free(tape);

    @memset(tape, 1.0);

    const self: *Self = try Self.init(page, tape, test_n);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 2);
    defer page.free(dest);

    var xvec: [1]f64 = .{test_x};

    self.forward(&xvec, test_mode, test_scale);

    self.cdata.backward(dest);
    self.scale.backward(dest);

    std.debug.print("value: {d}\nderiv: {d}\n", .{ self.value, dest });

    try testing.expectApproxEqRel(self.value[0], 0x1.73e2cedcce341p-3, 1e-16);
    try testing.expectApproxEqRel(dest[0], 0x1.a6eab34c5bb88p-7, 2e-16);
    try testing.expectApproxEqRel(dest[1], -0x1.4e6fd2407e1bdp-4, 1e-16);
}

const test_n: comptime_int = 1;
const test_x: comptime_float = 1.213;
const test_mode: comptime_float = 0.878;
const test_scale: comptime_float = 2.171;

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;

const CenteredData = @import("./CenteredData.zig");
const GaussianScale = @import("./GaussianScale.zig");
