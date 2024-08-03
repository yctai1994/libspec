//! Gaussian Mode
value: f64, // μ
deriv_in: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

const Self: type = @This(); // hosted by CenteredData

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 2 * n;
    if (tape.len != m + 1) unreachable;

    const self = try allocator.create(Self);
    self.deriv_in = tape[n..m]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    return self;
}

pub inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
}

pub inline fn forward(self: *Self, mode: f64) void {
    self.value = mode;
}

pub fn backward(self: *Self, deriv_out: []f64) void {
    if (deriv_out.len != 2) unreachable; // [ dy/dμ, dy/dσ ]

    // (dy/dμ) = [ dx̄₁/dμ, dx̄₂/dμ, … ]ᵀ⋅[ dy/dx̄₁, dy/dx̄₂, … ]
    // where x̄ᵢ ≡ xᵢ - μ ⇒ dx̄₁/dμ = dx̄₂/dμ = … = -1
    var temp: f64 = 0.0;
    for (self.deriv_in) |din| temp -= din;
    deriv_out[0] = temp;
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
