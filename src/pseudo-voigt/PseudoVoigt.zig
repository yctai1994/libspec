//! Pseudo-Voigt Function
value: []f64 = undefined, // [ Pv₁, Pv₂, … ]
deriv: []f64 = undefined, // [ dlogPv₁/dPv₁, dlogPv₂/dPv₂, … ]
deriv_in: []f64 = undefined, // [ dy/dlogPv₁, dy/dlogPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]

ratio: *PseudoVoigtRatio,
normal: *PseudoVoigtNormal,
lorentz: *PseudoVoigtLorentz,

const Self: type = @This(); // hosted by PseudoVoigtLogL

fn init(allocator: mem.Allocator, cdata: *CenteredData, width: *PseudoVoigtWidth, tape: []f64, n: usize) !*Self {
    if (tape.len != 5 * n + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.ratio = try PseudoVoigtRatio.init(allocator, width, tape, n);
    errdefer self.ratio.deinit(allocator);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.normal = try PseudoVoigtNormal.init(allocator, cdata, width, tape, n);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try PseudoVoigtLorentz.init(allocator, cdata, width, tape, n);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.deriv_in = tape[0..n]; // [ dy/dlogPv₁, dy/dlogPv₂, … ]
    self.deriv_out = tape[n .. n + n]; // [ dy/dPv₁, dy/dPv₂, … ]

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.lorentz.deinit(allocator);
    self.normal.deinit(allocator);
    self.ratio.deinit(allocator);

    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
}

fn forward(self: *Self) void {
    // PseudoVoigtWidth should be already forwarded.
    self.ratio.forward();
    self.normal.forward();
    self.lorentz.forward();

    const ratio: f64 = self.ratio.value;
    const complementary_ratio: f64 = 1.0 - ratio;
    for (self.value, self.normal.value, self.lorentz.value) |*pV, pN, pL| {
        pV.* = ratio * pL + complementary_ratio * pN;
    }

    for (self.normal.deriv, self.lorentz.deriv) |*dpN, *dpL| {
        dpN.* = complementary_ratio; // [ dPv₁/dPN₁, dPv₂/dPN₂, … ]
        dpL.* = ratio; // [ dPv₁/dPL₁, dPv₂/dPL₂, … ]
    }

    for (self.ratio.deriv, self.normal.value, self.lorentz.value) |*dratio, pN, pL| {
        dratio.* = pL - pN; // [ dPv₁/dη, dPv₂/dη, … ]
    }
}

fn backward(self: *Self) void {
    // [ dy/dPv₁, dy/dPv₂, … ] = [ dlogPv₁/dPv₁, dlogPv₂/dPv₂, … ]ᵀ ⋅ [ dy/dlogPv₁, dy/dlogPv₂, … ]
    for (self.deriv_out, self.deriv, self.deriv_in) |*dout, d, din| dout.* = d * din;

    self.ratio.backward();
    self.normal.backward();
    self.lorentz.backward();
}

test "PseudoVoigt: forward & backward" {
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

    cdata.forward(&xvec, test_mode);
    width.forward(test_sigma, test_gamma);
    self.forward();

    @memset(self.deriv, 1.0); // only need for unit-testing

    self.backward();
    cdata.backward(dest);
    width.backward(dest);

    try testing.expectApproxEqRel(0x1.e8dff2acbf52dp-4, self.value[0], 3e-16);
    try testing.expectApproxEqRel(0x1.80cfa7af6cc13p-8, dest[0], 6e-16);
    try testing.expectApproxEqRel(-0x1.15b8b266ad6e7p-5, dest[1], 9e-16);
    try testing.expectApproxEqRel(-0x1.12de1faa47042p-5, dest[2], 9e-16);
}

const test_n: comptime_int = 1;
const test_x: comptime_float = 1.213;
const test_mode: comptime_float = 0.878;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
const PseudoVoigtRatio = @import("./PseudoVoigtRatio.zig");
const PseudoVoigtNormal = @import("./PseudoVoigtNormal.zig");
const PseudoVoigtLorentz = @import("./PseudoVoigtLorentz.zig");
