//! Pseudo-Voigt Mode
value: f64 = undefined, // μ
deriv: []f64 = undefined, // [ dx̄₁/dμ, dx̄₂/dμ, … ], where x̄ᵢ ≡ xᵢ - μ
deriv_in: []f64 = undefined, // [ dy/dx̄₁, dy/dx̄₂, … ]

const Self: type = @This();

fn init(allocator: mem.Allocator, n: usize, tape: []f64) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);

    @memset(self.deriv, -1.0);

    // self.deriv_in = tape[TBD]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    _ = tape;

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
    return;
}

inline fn forward(self: *Self, mode: f64) void {
    self.value = mode;
    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

    // (dy/dμ) = [ dx̄₁/dμ, dx̄₂/dμ, … ]ᵀ ⋅ [ dy/dx̄₁, dy/dx̄₂, … ]
    final_deriv_out[0] = 0.0;
    for (self.deriv, self.deriv_in) |deriv, deriv_in| {
        final_deriv_out[0] += deriv * deriv_in;
    }

    return;
}

const std = @import("std");
const mem = std.mem;
