//! Centered Data
value: []f64, // [ x̄₁, x̄₂, … ], where x̄ᵢ ≡ xᵢ - μ
deriv: []f64, // [ dPN₁/dx̄₁, dPN₂/dx̄₂, … ]
deriv_in: []f64, // [ dy/dPN₁, dy/dPN₂, … ]
deriv_out: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

mode: *GaussianMode,

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

    self.mode = try GaussianMode.init(allocator, tape, n);

    self.deriv_in = tape[0..n]; // [ dy/dPN₁, dy/dPN₂, … ]
    self.deriv_out = tape[n..m]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.mode.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
}

pub fn forward(self: *Self, xvec: []f64, mode: f64) void {
    self.mode.forward(mode);
    for (self.value, xvec) |*cdat, data| cdat.* = data - mode;
}

pub fn backward(self: *Self, final_deriv_out: []f64) void {
    // [ dy/dx̄₁, dy/dx̄₂, … ] = [ dPN₁/dx̄₁, dPN₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPN₁, dy/dPN₂, … ]
    for (self.deriv_out, self.deriv, self.deriv_in) |*dout, d, din| dout.* = d * din;

    return self.mode.backward(final_deriv_out);
}

test "init" {
    const page = testing.allocator;
    var tape: [2 * test_n + 1]f64 = undefined;
    const self = try Self.init(page, &tape, test_n);
    defer self.deinit(page);
}

const test_n: comptime_int = 1;

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const GaussianMode = @import("./GaussianMode.zig");
