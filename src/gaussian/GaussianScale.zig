//! Gaussian Scale
value: f64, // σ
deriv: []f64, // [ dPN₁/dσ, dPN₂/dσ, … ]
deriv_in: []f64, // [ dy/dPN₁, dy/dPN₂, … ]

const Self: type = @This(); // hosted by

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 2 * n;
    if (tape.len != m + 1) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);

    self.deriv_in = tape[0..n]; // [ dy/dPN₁, dy/dPN₂, … ]

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
}

pub inline fn forward(self: *Self, scale: f64) void {
    self.value = scale;
}

pub fn backward(self: *Self, deriv_out: []f64) void {
    if (deriv_out.len != 2) unreachable; // [ dy/dμ, dy/dσ ]

    // (dy/dσ) = [ dPN₁/dσ, dPN₂/dσ, … ]ᵀ⋅[ dy/dPN₁, dy/dPN₂, … ]
    var temp: f64 = 0.0;
    for (self.deriv, self.deriv_in) |d, din| temp += d * din;
    deriv_out[1] = temp;
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
