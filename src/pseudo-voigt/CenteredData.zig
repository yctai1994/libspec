//! Centered Data
value: []f64, // [ x̄₁, x̄₂, … ], where x̄ᵢ ≡ xᵢ - μ
deriv: []f64, // [ dPN₁/dx̄₁, dPN₂/dx̄₂, …, dPL₁/dx̄₁, dPL₂/dx̄₂, … ]
deriv_in: []f64, // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
deriv_out: []f64, // [ dy/dx̄₁, dy/dx̄₂, … ]

mode: *PseudoVoigtMode,

const Self: type = @This(); // hosted by PseudoVoigtLogL

pub fn init(allocator: mem.Allocator, tape: []f64, n: usize) !*Self {
    const m: usize = 4 * n;
    if (tape.len != m + n + 6) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, 2 * n);
    errdefer allocator.free(self.deriv);

    self.mode = try PseudoVoigtMode.init(allocator, tape, n);

    self.deriv_in = tape[m >> 1 .. m]; // [ dy/dPN₁, dy/dPN₂, …, dy/dPL₁, dy/dPL₂, … ]
    self.deriv_out = tape[m .. m + n]; // [ dy/dx̄₁, dy/dx̄₂, … ]

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
    const n: usize = self.value.len;

    // [ dy/dx̄₁, dy/dx̄₂, … ] = [ dPN₁/dx̄₁, dPN₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPN₁, dy/dPN₂, … ]
    for (self.deriv_out, self.deriv[0..n], self.deriv_in[0..n]) |*dout, d, din| dout.* = d * din;

    // [ dy/dx̄₁, dy/dx̄₂, … ] += [ dPL₁/dx̄₁, dPL₂/dx̄₂, … ]ᵀ ⋅ [ dy/dPL₁, dy/dPL₂, … ]
    for (self.deriv_out, self.deriv[n..], self.deriv_in[n..]) |*dout, d, din| dout.* += d * din;

    return self.mode.backward(final_deriv_out);
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

const PseudoVoigtMode = @import("./PseudoVoigtMode.zig");
