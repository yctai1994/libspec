//! Pseudo-Voigt Mode
value: f64, // μ
// deriv: []f64,
// [ dx̄₁/dμ, dx̄₂/dμ, … ], where x̄ᵢ ≡ xᵢ - μ ⇒ dx̄₁/dμ = dx̄₂/dμ = … = -1
deriv_in: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

const Self: type = @This(); // hosted by CenteredData

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 4 * n;
    if (tape.len != m + n + 6) unreachable;

    const self = try allocator.create(Self);
    self.deriv_in = tape[m .. m + n]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    return self;
}

pub inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
}

// Called by CenteredData
pub inline fn forward(self: *Self, mode: f64) void {
    self.value = mode;
}

// Called by CenteredData
pub fn backward(self: *Self, deriv_out: []f64) void {
    if (deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

    // (dy/dμ) = [ dx̄₁/dμ, dx̄₂/dμ, … ]ᵀ⋅[ dy/dx̄₁, dy/dx̄₂, … ], dx̄₁/dμ = dx̄₂/dμ = … = -1
    deriv_out[0] = 0.0;
    for (self.deriv_in) |din| deriv_out[0] -= din;
}

test "init" {
    const page = testing.allocator;
    var tape: [6]f64 = undefined;
    const self = try Self.init(page, &tape, 0);
    defer self.deinit(page);
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
