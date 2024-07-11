//! Pseudo-Voigt Normal Scale
value: f64 = undefined, // σᵥ
deriv: []f64 = undefined, // [ dPN₁/dσᵥ, dPN₂/dσᵥ, … ]
deriv_in: []f64 = undefined, // [ dy/dPN₁, dy/dPN₂, … ]
deriv_out: *f64 = undefined, // dy/dσᵥ

width: *PseudoVoigtWidth, // hosted by PseudoVoigtLogL

const Self: type = @This(); // hosted by PseudoVoigtNormal

pub fn init(
    allocator: mem.Allocator,
    width: *PseudoVoigtWidth,
    n: usize,
    tape: []f64,
) !*Self {
    if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.deriv = try allocator.alloc(f64, n);
    self.width = width;

    // self.deriv_in = tape[TBD]; // [ dy/dPN₁, dy/dPN₂, … ]
    // self.deriv_out = &tape[TBD]; // dy/dσᵥ

    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.free(self.deriv);
    allocator.destroy(self);
    return;
}

fn forward(self: *Self) void {
    // Gamma should be already forwarded by `PseudoVoigt.preforward()`.
    // const temp: comptime_float = comptime 0.5 / @sqrt(2.0 * @log(2.0));
    // self.width.deriv[0] = temp; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
    // self.value = temp * self.width.value;
    _ = self;

    return;
}

fn backward(self: *Self) void {
    // (dy/dσV) = (dPN/dσV) × (dy/dPN)
    self.deriv_out.* = self.deriv * self.deriv_in.*;

    return;
}

const std = @import("std");
const mem = std.mem;

const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
