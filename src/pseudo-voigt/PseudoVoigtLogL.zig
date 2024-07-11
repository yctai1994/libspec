//! Pseudo-Voigt Log-Likelihood
tape: []f64,
value: f64, // logL
deriv_out: []f64, // [ dy/dlogPv₁, dy/dlogPv₂, … ]
// Note: dy/dlogPvᵢ = wᵢ when y = logL

width: *PseudoVoigtWidth,
cdata: *CenteredData,
pvoigt: *PseudoVoigt,

const Self: type = @This();

fn init(allocator: mem.Allocator, xvec: []f64) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    const n: usize = xvec.len;

    self.tape = try allocator.alloc(f64, 10);
    errdefer allocator.free(self.tape);

    self.width = try PseudoVoigtWidth.init(allocator, self.tape);
    errdefer self.width.deinit(allocator);

    self.cdata = try CenteredData.init(allocator, n, self.tape);
    errdefer self.cdata.deinit(allocator);

    self.pvoigt = try PseudoVoigt.init(allocator, self.width, self.cdata, n, self.tape);

    @memset(self.tape, 1.0);

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.pvoigt.deinit(allocator);
    self.cdata.deinit(allocator);
    self.width.deinit(allocator);

    allocator.free(self.tape);
    allocator.destroy(self);

    return;
}

fn forward(self: *Self) void {
    _ = self;
    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]
    _ = self;
    return;
}

const std = @import("std");
const mem = std.mem;

const PseudoVoigt = @import("./PseudoVoigt.zig");
const CenteredData = @import("./CenteredData.zig");
const PseudoVoigtWidth = @import("./PseudoVoigtWidth.zig");
