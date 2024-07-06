//! Centered Data
value: []f64 = undefined, // [ x̄₁, x̄₂, …, x̄ₙ ], where x̄ᵢ ≡ xᵢ - μ
deriv: []f64 = undefined, // [ dPN₁/dx̄₁, dPN₂/dx̄₂, …, dPL₁/dx̄₁, dPL₂/dx̄₂, … ]
deriv_in: []f64 = undefined, // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
deriv_out: []f64 = undefined, // [ dy/dx̄₁, dy/dx̄₂, … ]

const Self: type = @This();

fn init(allocator: mem.Allocator, n: usize, tape: []f64) !*Self {
    if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    defer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, 2 * n);

    // self.deriv_in = tape[TBD]; // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    return self;
}

inline fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
    return;
}

inline fn forward(self: *Self, xvec: f64) void {
    self.value = xvec;
    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    const n: usize = self.value.len;

    // [ dy/dx̄₁, dy/dx̄₂, … ] = [ dPN₁/dx̄₁, dPN₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPN₁, dy/dPN₂, … ]
    for (self.deriv[0..n], self.deriv_in[0..n], self.deriv_out) |deriv, deriv_in, *deriv_out| {
        deriv_out.* = deriv * deriv_in;
    }

    // [ dy/dx̄₁, dy/dx̄₂, … ] = [ dPL₁/dx̄₁, dPL₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPL₁, dy/dPL₂, … ]
    for (self.deriv[n..], self.deriv_in[n..], self.deriv_out) |deriv, deriv_in, *deriv_out| {
        deriv_out.* += deriv * deriv_in;
    }

    _ = final_deriv_out;

    return;
}

const std = @import("std");
const mem = std.mem;
