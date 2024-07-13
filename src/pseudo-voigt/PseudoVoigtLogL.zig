//! Pseudo-Voigt Log-Likelihood
tape: []f64,
value: f64, // logL
deriv_out: []f64, // [ dy/dlogPv₁, dy/dlogPv₂, … ]

width: *PseudoVoigtWidth,
cdata: *CenteredData,
pvoigt: *PseudoVoigt,

const Self: type = @This();

fn init(allocator: mem.Allocator, wvec: []f64) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    const n: usize = wvec.len;

    self.tape = try allocator.alloc(f64, 5 * n + 6);
    errdefer allocator.free(self.tape);

    self.width = try PseudoVoigtWidth.init(allocator, self.tape, n);
    errdefer self.width.deinit(allocator);

    self.cdata = try CenteredData.init(allocator, self.tape, n);
    errdefer self.cdata.deinit(allocator);

    self.pvoigt = try PseudoVoigt.init(allocator, self.cdata, self.width, self.tape, n);

    @memset(self.tape[n..], 1.0);

    self.deriv_out = self.tape[0..n];
    for (self.deriv_out, wvec) |*p, v| p.* = v; // dy/dlogPvᵢ = wᵢ when y = logL

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.pvoigt.deinit(allocator);
    self.cdata.deinit(allocator);
    self.width.deinit(allocator);

    allocator.free(self.tape);
    allocator.destroy(self);
}

fn forward(self: *Self, xvec: []f64, mode: f64, sigma: f64, gamma: f64) void {
    self.cdata.forward(xvec, mode);
    self.width.forward(sigma, gamma);
    self.pvoigt.forward();

    self.value = 0.0;
    for (self.deriv_out, self.pvoigt.deriv, self.pvoigt.value) |w, *dpV, pV| {
        self.value += w * @log(pV);
        dpV.* = 1.0 / pV; // [ dlogPv₁/dPv₁, dlogPv₂/dPv₂, … ]
    }
}

fn backward(self: *Self, deriv_out: []f64) void {
    self.pvoigt.backward();
    self.cdata.backward(deriv_out);
    self.width.backward(deriv_out);
}

test "PseudoVoigtLogL: forward & backward" {
    const page = testing.allocator;

    const xvec: []f64 = try page.alloc(f64, 21);
    defer page.free(xvec);

    const wvec: []f64 = try page.alloc(f64, 21);
    defer page.free(wvec);

    for (xvec, 0..) |*p, i| p.* = @as(f64, @floatFromInt(i)) - 10.0;

    // Generated exact Voigt function with (μ, σ, γ) = (-1.0, 1.53, 1.02)
    inline for (.{
        0x1.1cf5be03beea0p-8,
        0x1.71bc643f76115p-8,
        0x1.f657f84e55806p-8,
        0x1.6d5ba458e6b91p-7,
        0x1.24e2e3804fc37p-6, // 05
        0x1.04421cccb1950p-5,
        0x1.e14cf4fce3a26p-5,
        0x1.9cd1e2a6d554fp-4,
        0x1.28a1a44b3b02bp-3,
        0x1.50c5d694e3f72p-3, // 10
        0x1.28a1a44b3b02bp-3,
        0x1.9cd1e2a6d554fp-4,
        0x1.e14cf4fce3a26p-5,
        0x1.04421cccb1950p-5,
        0x1.24e2e3804fc37p-6, // 15
        0x1.6d5ba458e6b91p-7,
        0x1.f657f84e55806p-8,
        0x1.71bc643f76115p-8,
        0x1.1cf5be03beea0p-8,
        0x1.c5f095d8a3491p-9, // 20
        0x1.72b121352ac63p-9,
    }, wvec) |v, *p| p.* = v;

    const self: *Self = try Self.init(page, wvec);
    defer self.deinit(page);

    const dest: []f64 = try page.alloc(f64, 3);
    defer page.free(dest);

    self.forward(xvec, test_mode, test_sigma, test_gamma);
    self.backward(dest);

    try testing.expectApproxEqRel(-0x1.46f9bbe34e92fp+1, self.value, 4e-16);
    try testing.expectApproxEqRel(-0x1.55430572f2e75p-3, dest[0], 2e-16);
    try testing.expectApproxEqRel(0x1.ec76cbe9ae12dp-8, dest[1], 9e-15);
    try testing.expectApproxEqRel(-0x1.ff4ac03304d78p-5, dest[2], 8e-16);
}

const test_mode: comptime_float = 0.878;
const test_sigma: comptime_float = 2.171;
const test_gamma: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigt = @import("./PseudoVoigt.zig");
const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
