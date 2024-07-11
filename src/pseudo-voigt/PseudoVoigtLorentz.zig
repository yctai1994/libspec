// Pseudo-Voigt Lorentz
value: []f64 = undefined, // [ L(x̄₁, γᵥ), L(x̄₂, γᵥ), … ]
deriv: []f64 = undefined, // [ dPv₁/dPL₁, dPv₂/dPL₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPL₁, dy/dPL₂, … ]

cdata: *CenteredData, // hosted by PseudoVoigtLogL
scale: *PseudoLorentzScale,

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

    self.scale = try PseudoLorentzScale.init(allocator, width, tape);
    self.cdata = cdata;

    // self.deriv_in = tape[TBD]; // [ dy/dPv₁, dy/dPv₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dPL₁, dy/dPL₂, … ]

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
    const neg_twopi: comptime_float = comptime -2.0 * math.pi;
    const inv_width: f64 = 1.0 / self.scale.value;
    var temp: f64 = undefined;

    for (
        self.value,
        self.cdata.value,
        self.scale.deriv,
        self.cdata.deriv[n..],
    ) |prob, centered_x, *dwidth, *dcentered_x| {
        temp = neg_twopi * pow2(prob);
        dwidth.* = inv_width * prob - temp; // dPLᵢ/dγᵥ
        dcentered_x.* = inv_width * centered_x * temp; // dPLᵢ/dx̄ᵢ
    }

    return;
}

// Called by PseudoVoigt
fn backward(self: *Self) void {
    // [ dy/dPL₁, dy/dPL₂, … ] = [ dPv₁/dPL₁, dPv₂/dPL₂, … ]ᵀ ⋅ [ dy/dPv₁, dy/dPv₂, … ]
    for (self.deriv, self.deriv_in, self.deriv_out) |d, din, *dout| dout.* = d * din;
    return;
}

inline fn density(centered_x: f64, width: f64) f64 {
    const temp: comptime_float = comptime 1.0 / math.pi;
    return temp / (width * (1.0 + pow2(centered_x / width)));
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
const PseudoLorentzScale = @import("./PseudoLorentzScale.zig");
