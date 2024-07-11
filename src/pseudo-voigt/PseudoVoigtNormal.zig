//! Pseudo-Voigt Normal
value: []f64 = undefined, // [ N(x̄₁, σᵥ), N(x̄₂, σᵥ), … ]
deriv: []f64 = undefined, // [ dPv₁/dPN₁, dPv₂/dPN₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPN₁, dy/dPN₂, … ]

cdata: *CenteredData, // hosted by PseudoVoigtLogL
scale: *PseudoNormalScale,

const Self: type = @This(); // hosted by PseudoVoigt

// Called by PseudoVoigt
fn init(
    allocator: mem.Allocator,
    width: *PseudoVoigtWidth,
    cdata: *CenteredData,
    n: usize,
    tape: []f64,
) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    self.scale = try PseudoNormalScale.init(allocator, width, tape);
    self.cdata = cdata;

    // self.deriv_in = tape[TBD]; // [ dy/dPv₁, dy/dPv₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dPN₁, dy/dPN₂, … ]

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
    for (self.value, self.cdata.value) |*prob, centered_x| {
        prob.* = density(centered_x, self.scale.value);
    }

    const n: usize = self.value.len;
    const inv_sigma: f64 = 1.0 / self.scale.value;
    // arg1ᵢ = x̄ᵢ/σᵥ, arg2ᵢ = PNᵢ/σᵥ
    for (
        self.deriv,
        self.deriv_out,
        self.cdata.value,
        self.value,
    ) |*arg1, *arg2, centered_x, prob| {
        arg1.* = inv_sigma * centered_x;
        arg2.* = inv_sigma * prob;
    }

    for (
        self.deriv,
        self.deriv_out,
        self.scale.deriv,
        self.cdata.deriv[0..n],
    ) |arg1, arg2, *dsigma, *dcentered_x| {
        dsigma.* = arg1 * arg2; // dPNᵢ/dσᵥ ← (arg1ᵢ)⋅(arg2ᵢ)
        dcentered_x.* = -dsigma.*; // dPNᵢ/dx̄ᵢ = -(arg1ᵢ)⋅(arg2ᵢ)
    }

    for (
        self.deriv,
        self.deriv_out,
        self.scale.deriv,
    ) |arg1, arg2, *dsigma| {
        dsigma.* = dsigma.* * arg1 - arg2; // dPNᵢ/dσᵥ = (arg1ᵢ)²⋅(arg2ᵢ) - arg2ᵢ
    }

    return;
}

// Called by PseudoVoigt
fn backward(self: *Self) void {
    // [ dy/dPN₁, dy/dPN₂, … ] = [ dPv₁/dPN₁, dPv₂/dPN₂, … ]ᵀ⋅[ dy/dPv₁, dy/dPv₂, … ]
    for (self.deriv, self.deriv_in, self.deriv_out) |d, din, *dout| dout.* = d * din;
    return;
}

inline fn density(centered_x: f64, width: f64) f64 {
    const temp: comptime_float = comptime 1.0 / @sqrt(2.0 * math.pi);
    return temp * @exp(-0.5 * pow2(centered_x / width)) / width;
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
const PseudoNormalScale = @import("./PseudoNormalScale.zig");
