// Pseudo-Voigt Lorentz
value: []f64 = undefined, // [ L(x̄₁, γᵥ), L(x̄₂, γᵥ), … ]
deriv: []f64 = undefined, // [ dPv₁/dPL₁, dPv₂/dPL₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPL₁, dy/dPL₂, … ]

cdata: *CenteredData, // handled by PseudoVoigt
scale: *PseudoLorentzScale,

const Self: type = @This();

fn init(
    allocator: mem.Allocator,
    gamma: *PseudoVoigtGamma,
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

    self.scale = try PseudoLorentzScale.init(allocator, gamma, tape);
    self.cdata = cdata;

    // self.deriv_in = tape[TBD]; // [ dy/dPv₁, dy/dPv₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dPL₁, dy/dPL₂, … ]

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
    return;
}

fn forward(self: *Self) void {
    for (self.value, self.cdata.value) |*prob, data| {
        prob.* = density(data, self.scale.value);
    }

    // const arg1: f64 = (x - self.mode.value) / pow2(self.scale.value);
    // const arg2: f64 = self.scale.value * pow2(arg1) - 1.0 / self.scale.value;

    // self.value = prob;
    // self.mode.deriv[0] = prob * arg1; // [ dPN/dμ, dPL/dμ ]
    // self.scale.deriv = prob * arg2; // dPN/dσV

    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    // [ dy/dPL₁, dy/dPL₂, … ] = [ dPv₁/dPL₁, dPv₂/dPL₂, … ]ᵀ ⋅ [ dy/dPv₁, dy/dPv₂, … ]
    for (self.deriv, self.deriv_in, self.deriv_out) |deriv, deriv_in, *deriv_out| {
        deriv_out.* = deriv * deriv_in;
    }

    _ = final_deriv_out;

    return;
}

inline fn density(x: f64, gamma: f64) f64 {
    const temp: comptime_float = comptime 1.0 / math.pi;
    return temp / (gamma * (1.0 + pow2(x / gamma)));
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtGamma = @import("./PseudoVoigtGamma.zig");
const PseudoLorentzScale = @import("./PseudoLorentzScale.zig");
