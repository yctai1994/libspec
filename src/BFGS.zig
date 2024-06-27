//! References
//! [1] J. E. Dennis, R. B. Schnabel,
//!     "Numerical Methods for Unconstrained Optimization and Nonlinear Equations,"
//!     1993, Sec. 3.4
//! [2] William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery,
//!     "Numerical Recipes 3rd Edition: The Art of Scientific Computing,"
//!     2007, Sec. 2.10.1

n: usize, // dims
w: []f64, // this is a buffer
R: [][]f64,

const Erros = error{SingularError};

fn update(self: *const @This(), u: []f64, v: []f64) !void {
    var k: usize = 0;
    // Find largest k such that u[k] ≠ 0.
    for (self.w, u, 0..) |*w_k, u_k, i| {
        if (u_k != 0.0) k = i;
        w_k.* = u_k;
    }

    // Transform R + u⋅vᵀ to upper Hessenberg.
    var i = k - 1;
    while (0 <= i) : (i -= 1) {
        self.rotate(i, self.w[i], -self.w[i + 1]);
        if (self.w[i] == 0.0) {
            self.w[i] = @abs(self.w[i + 1]);
        } else if (@abs(self.w[i]) > @abs(self.w[i + 1])) {
            self.w[i] = @abs(self.w[i]) * @sqrt(1.0 + pow2(f64, self.w[i + 1] / self.w[i]));
        } else {
            self.w[i] = @abs(self.w[i + 1]) * @sqrt(1.0 + pow2(f64, self.w[i] / self.w[i + 1]));
        }

        if (i == 0) break;
    }

    for (self.R[0], v) |*R_0i, v_i| R_0i.* += self.w[0] * v_i;

    // Transform upper Hessenberg matrix to upper triangular.
    for (0..k) |j| {
        self.rotate(j, self.R[j][j], -self.R[j + 1][j]);
    }

    for (self.R, 0..self.n) |R_j, j| if (R_j[j] == 0.0) return error.SingularError;

    return;
}

fn rotate(self: *const @This(), i: usize, a: f64, b: f64) void {
    var c: f64 = undefined;
    var s: f64 = undefined;
    var w: f64 = undefined;
    var y: f64 = undefined;
    var f: f64 = undefined;

    if (a == 0.0) {
        c = 0.0;
        s = if (b < 0.0) -1.0 else 1.0;
    } else if (@abs(a) > @abs(b)) {
        f = b / a;
        c = math.copysign(1.0 / @sqrt(1.0 + (f * f)), a);
        s = f * c;
    } else {
        f = a / b;
        s = math.copysign(1.0 / @sqrt(1.0 + (f * f)), b);
        c = f * s;
    }

    for (
        self.R[i][i..self.n],
        self.R[i + 1][i..self.n],
    ) |*R_ij, *R_ip1j| {
        y = R_ij.*;
        w = R_ip1j.*;
        R_ij.* = c * y - s * w;
        R_ip1j.* = s * y + c * w;
    }

    return;
}

test "BFGS.update" {
    const ArrF64 = Array(f64){ .allocator = std.testing.allocator };
    const R: [][]f64 = try ArrF64.matrix(3, 3);
    defer ArrF64.free(R);

    inline for (.{ 2.0, 6.0, 8.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, R[2]) |val, *ptr| ptr.* = val;

    const u: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(u);

    const v: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(v);

    inline for (.{ 1.0, 5.0, 3.0 }, u) |val, *ptr| ptr.* = val;
    inline for (.{ 2.0, 3.0, 1.0 }, v) |val, *ptr| ptr.* = val;

    const w: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(w);

    const bfgs: @This() = .{ .n = 3, .w = w, .R = R };
    try bfgs.update(u, v);

    const A: [3][3]f64 = .{ // answers
        .{ 0x1.8a85c24f70658p+03, 0x1.44715e1c46896p+04, 0x1.be6ef01685ec3p+03 },
        .{ -0x1.0000000000000p-52, 0x1.4e2ba31c14a89p+01, 0x1.28c0f1b618468p+02 },
        .{ 0x0.0000000000000p+00, 0x0.0000000000000p+00, -0x1.dd36445718509p-01 },
    };

    for (A, R) |A_i, R_i| {
        try testing.expect(std.mem.eql(f64, &A_i, R_i));
    }
}

const std = @import("std");
const math = std.math;
const testing = std.testing;
const pow2 = @import("./poly.zig").pow2;
const Array = @import("./array.zig").Array;
