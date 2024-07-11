//! Pseudo-Voigt Function
value: []f64 = undefined, // [ Pv₁, Pv₂, … ]
deriv: []f64 = undefined, // [ dlogPv₁/dPv₁, dlogPv₂/dPv₂, … ]
deriv_in: []f64 = undefined, // [ dy/dlogPv₁, dy/dlogPv₂, … ]
deriv_out: []f64 = undefined, // [ dy/dPv₁, dy/dPv₂, … ]

eta: *PseudoVoigtEta,
normal: *PseudoVoigtNormal,
lorentz: *PseudoVoigtLorentz,

const Self: type = @This(); // hosted by PseudoVoigtLogL

fn init(
    allocator: mem.Allocator,
    width: *PseudoVoigtWidth,
    cdata: *CenteredData,
    n: usize,
    tape: []f64,
) !*Self {
    // if (tape.len != 10) unreachable;

    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.value = try allocator.alloc(f64, n);
    errdefer allocator.free(self.value);

    self.deriv = try allocator.alloc(f64, n);
    errdefer allocator.free(self.deriv);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.eta = try PseudoVoigtEta.init(allocator, width, tape);
    errdefer self.eta.deinit(allocator);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.normal = try PseudoVoigtNormal.init(allocator, width, cdata, n, tape);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try PseudoVoigtLorentz.init(allocator, width, cdata, n, tape);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    // self.deriv_in = tape[TBD]; // [ dy/dlogPv₁, dy/dlogPv₂, … ]
    // self.deriv_out = tape[TBD]; // [ dy/dPv₁, dy/dPv₂, … ]

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.lorentz.deinit(allocator);
    self.normal.deinit(allocator);
    self.eta.deinit(allocator);

    allocator.free(self.deriv);
    allocator.free(self.value);
    allocator.destroy(self);

    return;
}

fn forward(self: *Self) void {
    _ = self;
    return;
}

fn backward(self: *Self) void {
    // [ dy/dPv₁, dy/dPv₂, … ] = [ dlogPv₁/dPv₁, dlogPv₂/dPv₂, … ]ᵀ ⋅ [ dy/dlogPv₁, dy/dlogPv₂, … ]
    for (self.deriv, self.deriv_in, self.deriv_out) |d, din, *dout| dout.* = d * din;
    return;
}

const std = @import("std");
const mem = std.mem;

const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
const PseudoVoigtEta = @import("./PseudoVoigtEta.zig");
const PseudoVoigtNormal = @import("./PseudoVoigtNormal.zig");
const PseudoVoigtLorentz = @import("./PseudoVoigtLorentz.zig");
