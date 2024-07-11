//! Centered Data
value: []f64, // [ x̄₁, x̄₂, … ], where x̄ᵢ ≡ xᵢ - μ
deriv: []f64, // [ dPN₁/dx̄₁, dPN₂/dx̄₂, …, dPL₁/dx̄₁, dPL₂/dx̄₂, … ]
deriv_in: []f64, // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
deriv_out: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

mode: *PseudoVoigtMode,

const Self: type = @This(); // hosted by PseudoVoigtLogL

fn init(allocator: mem.Allocator, n: usize, tape: []f64) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, 2 * n);
    errdefer allocator.free(self.deriv);

    self.mode = try PseudoVoigtMode.init(allocator, tape);

    // self.deriv_in = tape[TBD]; // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dx̄₁, dy/dx̄₂, … ]

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.mode.deinit(allocator);
    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);
    return;
}

// Called by PseudoVoigtLogL
fn forward(self: *Self, xvec: f64, mode: f64) void {
    self.mode.forward(mode);
    for (self.value, xvec) |*cdat, data| cdat.* = data - mode;
    return;
}

// Called by PseudoVoigtLogL
fn backward(self: *Self, final_deriv_out: []f64) void {
    const n: usize = self.value.len;

    // [ dy/dx̄₁, dy/dx̄₂, … ] = [ dPN₁/dx̄₁, dPN₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPN₁, dy/dPN₂, … ]
    for (self.deriv[0..n], self.deriv_in[0..n], self.deriv_out) |d, din, *dout| dout.* = d * din;

    // [ dy/dx̄₁, dy/dx̄₂, … ] += [ dPL₁/dx̄₁, dPL₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPL₁, dy/dPL₂, … ]
    for (self.deriv[n..], self.deriv_in[n..], self.deriv_out) |d, din, *dout| dout.* += d * din;

    self.mode.backward(final_deriv_out);

    return;
}

test "init" {
    const page = testing.allocator;
    const self = try Self.init(page, 10, &.{});
    defer self.deinit(page);
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;

const PseudoVoigtMode = @import("./PseudoVoigtMode.zig");
