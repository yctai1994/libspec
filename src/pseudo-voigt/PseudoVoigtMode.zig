//! Pseudo-Voigt Mode
value: f64, // μ
// deriv: []f64,
// [ dx̄₁/dμ, dx̄₂/dμ, … ], where x̄ᵢ ≡ xᵢ - μ ⇒ dx̄₁/dμ = dx̄₂/dμ = … = -1
deriv_in: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

const Self: type = @This(); // hosted by CenteredData

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);

    // self.deriv_in = tape[TBD]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    _ = .{ tape, n };

    return self;
}

pub inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

// Called by CenteredData
pub inline fn forward(self: *Self, mode: f64) void {
    self.value = mode;
    return;
}

// Called by CenteredData
pub fn backward(self: *Self, final_deriv_out: []f64) void {
    if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

    // (dy/dμ) = [ dx̄₁/dμ, dx̄₂/dμ, … ]ᵀ⋅[ dy/dx̄₁, dy/dx̄₂, … ], dx̄₁/dμ = dx̄₂/dμ = … = -1
    final_deriv_out[0] = 0.0;
    for (self.deriv_in) |deriv_in| final_deriv_out[0] -= deriv_in;
    return;
}

test "init" {
    const page = testing.allocator;
    const self = try Self.init(page, &.{}, 0);
    defer self.deinit(page);
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
