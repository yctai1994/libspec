// Pseudo-Voigt Lorentz
value: []f64 = undefined, // [ L(x̄₁, γᵥ), L(x̄₂, γᵥ), … ]
deriv: []f64 = undefined, // [ dPv₁/dPL₁, dPv₂/dPL₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPL₁, dy/dPL₂, … ]

cdata: *CenteredData, // hosted by PseudoVoigtLogL
scale: *PseudoLorentzScale,

const Self: type = @This(); // hosted by PseudoVoigt

// Called by PseudoVoigt
pub fn init(allocator: mem.Allocator, cdata: *CenteredData, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    const m: usize = 2 * n;
    if (tape.len != (m <<| 1) + n + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    self.scale = try PseudoLorentzScale.init(allocator, width, tape, n);
    self.cdata = cdata;

    self.deriv_in = tape[n..m]; // [ dy/dPv₁, dy/dPv₂, … ]
    self.deriv_out = tape[m + n .. m <<| 1]; // [ dy/dPL₁, dy/dPL₂, … ]

    return self;
}

// Called by PseudoVoigt
pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.scale.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
    return;
}

// Called by PseudoVoigt
pub fn forward(self: *Self) void {
    self.scale.forward();

    for (self.value, self.cdata.value) |*prob, centered_x| {
        prob.* = density(centered_x, self.scale.value);
    }

    const n: usize = self.value.len;
    const neg_twopi: comptime_float = comptime -2.0 * math.pi;
    const inv_scale: f64 = 1.0 / self.scale.value;
    var temp: f64 = undefined;

    for (self.value, self.cdata.value, self.scale.deriv, self.cdata.deriv[n..]) |prob, centered_x, *dscale, *dcentered_x| {
        temp = neg_twopi * pow2(prob);
        dscale.* = inv_scale * prob + temp; // dPLᵢ/dγᵥ
        dcentered_x.* = inv_scale * centered_x * temp; // dPLᵢ/dx̄ᵢ
    }

    return;
}

// Called by PseudoVoigt
pub fn backward(self: *Self) void {
    // [ dy/dPL₁, dy/dPL₂, … ] = [ dPv₁/dPL₁, dPv₂/dPL₂, … ]ᵀ ⋅ [ dy/dPv₁, dy/dPv₂, … ]
    for (self.deriv_out, self.deriv, self.deriv_in) |*dout, d, din| dout.* = d * din;
    return self.scale.backward();
}

inline fn density(centered_x: f64, width: f64) f64 {
    const temp: comptime_float = comptime 1.0 / math.pi;
    return temp / (width * (1.0 + pow2(centered_x / width)));
}

fn pow2(x: f64) f64 {
    return x * x;
}

test "PseudoVoigtLorentz: forward & backward" {
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

    try testing.expectApproxEqRel(0x1.85dd27b58e542p-4, self.value[0], 2e-16);
    try testing.expectApproxEqRel(0x1.7984d5b740738p-8, dest[0], 5e-16);
    try testing.expectApproxEqRel(-0x1.069c4b385bde9p-5, dest[1], 9e-16);
    try testing.expectApproxEqRel(-0x1.28ffb243ea7d8p-6, dest[2], 2e-15);
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
const PseudoLorentzScale = @import("./PseudoLorentzScale.zig");
