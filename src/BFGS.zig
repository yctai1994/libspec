//! References
//! [1] J. E. Dennis, R. B. Schnabel,
//!     "Numerical Methods for Unconstrained Optimization and Nonlinear Equations,"
//!     1993, Sec. 3.4
//! [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
//!     "Numerical Recipes 3rd Edition: The Art of Scientific Computing,"
//!     2007, Sec. 2.10.1
//! [3] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Procedure. 18.2

xm: []f64 = undefined, // xₘ
xn: []f64 = undefined, // xₙ

gm: []f64 = undefined, // ∇f(xₘ)
gn: []f64 = undefined, // ∇f(xₙ)

sm: []f64 = undefined,
ym: []f64 = undefined,
am: []f64 = undefined,

um: []f64 = undefined,
vm: []f64 = undefined,

Bm: *Hess = undefined, // current approximation of Hessian
df: *const fn (x: []f64, g: []f64) void,

// already comptime scope
const slice_al: comptime_int = @alignOf([]f64);
const child_al: comptime_int = @alignOf(f64);
const slice_sz: comptime_int = @sizeOf(usize) * 2;
const child_sz: comptime_int = @sizeOf(f64);

const AllocError = mem.Allocator.Error;

pub fn init(self: *@This(), allocator: mem.Allocator, n: usize) AllocError!void {
    self.xm = try allocator.alloc(f64, n);
    errdefer allocator.free(self.xm);

    self.xn = try allocator.alloc(f64, n);
    errdefer allocator.free(self.xn);

    self.gm = try allocator.alloc(f64, n);
    errdefer allocator.free(self.gm);

    self.gn = try allocator.alloc(f64, n);
    errdefer allocator.free(self.gn);

    self.sm = try allocator.alloc(f64, n);
    errdefer allocator.free(self.sm);

    self.ym = try allocator.alloc(f64, n);
    errdefer allocator.free(self.ym);

    self.am = try allocator.alloc(f64, n);
    errdefer allocator.free(self.am);

    self.um = try allocator.alloc(f64, n);
    errdefer allocator.free(self.um);

    self.vm = try allocator.alloc(f64, n);
    errdefer allocator.free(self.vm);

    self.Bm = try allocator.create(Hess);
    errdefer allocator.destroy(self.Bm);

    try self.Bm.init(allocator, n);

    return;
}

pub fn deinit(self: *const @This(), allocator: mem.Allocator) void {
    allocator.free(self.xm);
    allocator.free(self.xn);
    allocator.free(self.gm);
    allocator.free(self.gn);
    allocator.free(self.sm);
    allocator.free(self.ym);
    allocator.free(self.am);
    allocator.free(self.um);
    allocator.free(self.vm);

    self.Bm.deinit(allocator);
    allocator.destroy(self.Bm);

    return;
}

test "BFGS.init and BFGS.deinit" {
    const page = std.testing.allocator;

    var bfgs: @This() = .{ .df = ellipse_deriv };
    try bfgs.init(page, 3);
    bfgs.deinit(page);
}

fn iterate(self: *const @This()) !void {
    // std.debug.print("xm = {e}\n", .{self.xm});

    // gₘ ← ∇f(xₘ)
    self.df(self.xm, self.gm);
    // std.debug.print("gm = {e}\n", .{self.gm});

    // pₘ ← Bₘ⁻¹⋅∇f(xₘ), store `pₘ` in `sₘ`
    // pₘ := sₘ, ∇f(xₘ) := gₘ
    self.Bm.solve(self.sm, self.gm);

    // xₙ ← xₘ + pₘ
    for (self.xn, self.xm, self.sm) |*xn_i, xm_i, pc_i| xn_i.* = xm_i - pc_i;
    // std.debug.print("xn = {e}\n", .{self.xn});

    // gₙ ← ∇f(xₙ)
    self.df(self.xn, self.gn);
    // std.debug.print("gn = {e}\n", .{self.gn});

    // sₘ ← xₙ - xₘ
    for (self.sm, self.xn, self.xm) |*sm_i, xn_i, xm_i| sm_i.* = xn_i - xm_i;
    // std.debug.print("sm = {e}\n", .{self.sm});

    // yₘ ← ∇f(xₙ) - ∇f(xₘ) = gₙ - gₘ
    for (self.ym, self.gn, self.gm) |*ym_i, gn_i, gm_i| ym_i.* = gn_i - gm_i;
    // std.debug.print("ym = {e}\n", .{self.ym});

    // secant_norm2 ← yₘᵀ⋅sₘ
    var secant_norm2: f64 = dot(self.ym, self.sm);
    // std.debug.print("secant_norm2 = {e}\n", .{secant_norm2});

    // sₘ ← Lₘᵀ⋅sₘ = Rₘ⋅sₘ
    self.Bm.dtrmv(self.sm);

    // sₘᵀ⋅(Lₘ⋅Lₘᵀ)⋅sₘ ← sₘᵀ⋅sₘ
    const quadratic_form: f64 = dot(self.sm, self.sm);
    // std.debug.print("quadratic_form = {e}\n", .{quadratic_form});

    // Damped BFGS parameter θₘ [3]
    //     rₘ = θₘ⋅yₘ + (1 - θₘ)⋅Bₘ⋅sₘ
    //     θₘ = | 1,                                    if yₘᵀ⋅sₘ ≥ 0.2⋅sₘᵀ⋅Bₘ⋅sₘ
    //          | 0.8⋅sₘᵀ⋅Bₘ⋅sₘ / (sₘᵀ⋅Bₘ⋅sₘ - yₘᵀ⋅sₘ), if yₘᵀ⋅sₘ < 0.2⋅sₘᵀ⋅Bₘ⋅sₘ
    // rₘᵀ⋅sₘ = | yₘᵀ⋅sₘ,                               if yₘᵀ⋅sₘ ≥ 0.2⋅sₘᵀ⋅Bₘ⋅sₘ
    //          | 0.2⋅sₘᵀ⋅Bₘ⋅sₘ,                        if yₘᵀ⋅sₘ < 0.2⋅sₘᵀ⋅Bₘ⋅sₘ
    var theta_m: f64 = undefined;
    if (0.2 * quadratic_form <= secant_norm2) {
        theta_m = 1.0;
    } else {
        theta_m = 0.8 * quadratic_form / (quadratic_form - secant_norm2);
        secant_norm2 = 0.2 * quadratic_form;
    }
    // std.debug.print("theta_m = {d}\n", .{theta_m});

    // αₘ ← √(secant_norm2 / quadratic_form)
    const alpha_m: f64 = @sqrt(secant_norm2 / quadratic_form);
    // std.debug.print("alpha_m = {e}\n", .{alpha_m});

    // aₘ ← αₘ⋅Lₘᵀ⋅sₘ = αₘ⋅Rₘ⋅sₘ
    for (self.am, self.sm) |*am_i, sm_i| am_i.* = alpha_m * sm_i;
    // std.debug.print("am = {e}\n", .{self.am});

    // ‖aₘ‖ ← √(rₘᵀ⋅sₘ)
    const secant_norm: f64 = @sqrt(secant_norm2);
    // std.debug.print("secant_norm = {e}\n", .{secant_norm});

    // uₘ ← aₘ / ‖aₘ‖
    for (self.um, self.am) |*um_i, am_i| um_i.* = am_i / secant_norm;
    // std.debug.print("um = {e}\n", .{self.um});

    // vₘ ← θₘ⋅yₘ
    for (self.vm, self.ym) |*vm_i, ym_i| vm_i.* = theta_m * ym_i;

    // vₘ ← vₘ + Rₘᵀ[(1 - θₘ - αₘ)⋅Rₘ⋅sₘ]
    const temp: f64 = 1.0 - theta_m - alpha_m;
    for (self.Bm.matrix, self.sm, 0..) |R_i, s_i, i| {
        for (self.vm[i..], R_i[i..]) |*v_j, R_ij| {
            v_j.* += temp * R_ij * s_i;
        }
    }

    // vₘ ← vₘ / ‖aₘ‖
    for (self.vm) |*vm_i| vm_i.* /= secant_norm;
    // std.debug.print("vm = {e}\n", .{self.vm});

    try self.Bm.update(self.um, self.vm);
    // std.debug.print("Bm = {e}\n", .{self.Bm.matrix});

    @memcpy(self.xm, self.xn); // copyto(self.xn, self.xm);
    @memcpy(self.gm, self.gn); // copyto(self.gn, self.gm);

    return;
}

test "BFGS.iterate #1" {
    // Case: Example 9.2.2 in [1]
    const page = std.testing.allocator;

    var bfgs: @This() = .{ .df = ellipse_deriv };
    try bfgs.init(page, 2);
    defer bfgs.deinit(page);

    inline for (.{ 1.0, 1.0 }, bfgs.xm) |v, *p| p.* = v;

    inline for (.{ 0x1.deeea11683f49p+1, -0x1.11acee560242ap+0 }, bfgs.Bm.matrix[0]) |v, *p| p.* = v;
    inline for (.{ 0x0.0000000000000p+0, 0x1.b0b80ef844ba1p+0 }, bfgs.Bm.matrix[1]) |v, *p| p.* = v;

    for (0..13) |_| try bfgs.iterate();
    // std.debug.print("{d}\n", .{bfgs.xm});

    const y: [2]f64 = .{ 2.0, -1.0 };
    try testing.expect(mem.eql(f64, &y, bfgs.xm));
}

test "BFGS.iterate #2" {
    // Case: Rosenbrock's function
    const page = std.testing.allocator;

    var bfgs: @This() = .{ .df = rosenbrock_deriv };
    try bfgs.init(page, 2);
    defer bfgs.deinit(page);

    inline for (.{ -1.2, 1.0 }, bfgs.xm) |v, *p| p.* = v;

    inline for (.{ 0x1.23c0d99c17436p+5, 0x1.a52d7f6fc5311p+3 }, bfgs.Bm.matrix[0]) |v, *p| p.* = v;
    inline for (.{ 0x0.0000000000000p+0, 0x1.4b1d7f7c3508bp+2 }, bfgs.Bm.matrix[1]) |v, *p| p.* = v;

    for (0..58) |_| try bfgs.iterate();
    // std.debug.print("{d}\n", .{bfgs.xm});

    const y: [2]f64 = .{ 1.0, 1.0 };
    try testing.expect(mem.eql(f64, &y, bfgs.xm));
}

const Hess = struct {
    buffer: []f64 = undefined,
    matrix: [][]f64 = undefined,

    const HessError = error{SingularError};

    fn init(self: *Hess, allocator: mem.Allocator, n: usize) AllocError!void {
        self.buffer = try allocator.alloc(f64, n);
        errdefer allocator.free(self.buffer);

        const temp: []u8 = try allocator.alloc(u8, n * n * child_sz + n * slice_sz);

        self.matrix = blk: {
            const ptr: [*]align(slice_al) []f64 = @ptrCast(@alignCast(temp.ptr));
            break :blk ptr[0..n];
        };

        const chunk_sz: usize = n * child_sz;
        var padding: usize = n * slice_sz;

        for (self.matrix) |*row| {
            row.* = blk: {
                const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(temp.ptr + padding));
                break :blk ptr[0..n];
            };
            padding += chunk_sz;
        }

        return;
    }

    fn deinit(self: *const Hess, allocator: mem.Allocator) void {
        const n: usize = self.buffer.len;
        const ptr: [*]u8 = @ptrCast(@alignCast(self.matrix.ptr));
        const len: usize = n * n * child_sz + n * slice_sz;

        allocator.free(self.buffer);
        allocator.free(ptr[0..len]);

        return;
    }

    fn update(self: *Hess, u: []f64, v: []f64) HessError!void {
        const n: usize = u.len;
        if (n != v.len) unreachable;

        // Find largest k such that u[k] ≠ 0.
        var k: usize = 0;
        for (self.buffer, u, 0..) |*ptr, val, index| {
            if (val != 0.0) k = index;
            ptr.* = val;
        }

        // Transform R + u⋅vᵀ to upper Hessenberg.
        var i: usize = k - 1;
        while (0 <= i) : (i -= 1) {
            rotate(self.matrix, i, n, self.buffer[i], -self.buffer[i + 1]);

            if (self.buffer[i] == 0.0) {
                self.buffer[i] = @abs(self.buffer[i + 1]);
            } else if (@abs(self.buffer[i]) > @abs(self.buffer[i + 1])) {
                self.buffer[i] = @abs(self.buffer[i]) * @sqrt(1.0 + pow2(f64, self.buffer[i + 1] / self.buffer[i]));
            } else {
                self.buffer[i] = @abs(self.buffer[i + 1]) * @sqrt(1.0 + pow2(f64, self.buffer[i] / self.buffer[i + 1]));
            }

            if (i == 0) break;
        }

        for (self.matrix[0], v) |*R_0i, v_i| {
            R_0i.* += self.buffer[0] * v_i;
        }

        // Transform upper Hessenberg matrix to upper triangular.
        for (0..k, 1..) |j, jp1| {
            rotate(self.matrix, j, n, self.matrix[j][j], -self.matrix[jp1][j]);
        }

        for (self.matrix, 0..n) |R_j, j| {
            if (R_j[j] == 0.0) return HessError.SingularError;
        }

        return;
    }

    fn rotate(R: [][]f64, i: usize, n: usize, a: f64, b: f64) void {
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

        for (R[i][i..n], R[i + 1][i..n]) |*R_ij, *R_ip1j| {
            y = R_ij.*;
            w = R_ip1j.*;
            R_ij.* = c * y - s * w;
            R_ip1j.* = s * y + c * w;
        }

        return;
    }

    // Cholesky
    fn solve(self: *const Hess, x: []f64, b: []f64) void {
        const n: usize = b.len;
        if (n != x.len) unreachable;

        var t: f64 = undefined;
        var i: usize = n - 1;
        var j: usize = undefined;

        // Rᵀ⋅y = b

        @memcpy(x, b); // copyto(b, x);

        for (self.matrix, x, 1..) |A_i, *x_i, ip1| {
            x_i.* /= A_i[ip1 - 1];
            for (A_i[ip1..n], x[ip1..n]) |A_ij, *x_j| {
                x_j.* -= A_ij * x_i.*;
            }
        }

        // R⋅x = y

        while (true) : (i -= 1) {
            t = x[i];
            j = i + 1;
            for (self.matrix[i][j..n], x[j..n]) |A_ij, x_j| {
                t -= A_ij * x_j;
            }
            x[i] = t / self.matrix[i][i];
            if (i == 0) break;
        }

        return;
    }

    fn dtrmv(self: *Hess, x: []f64) void {
        var temp: f64 = undefined;
        for (self.matrix, x, 0..) |A_i, *x_i, i| {
            temp = 0.0;
            for (A_i[i..], x[i..]) |A_ij, x_j| {
                temp += A_ij * x_j;
            }
            x_i.* = temp;
        }
        return;
    }
};

test "Hess.update" {
    const page = std.testing.allocator;

    var hess: Hess = .{};
    try hess.init(page, 3);
    defer hess.deinit(page);

    inline for (.{ 2.0, 6.0, 8.0 }, hess.matrix[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, hess.matrix[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, hess.matrix[2]) |val, *ptr| ptr.* = val;

    const u: []f64 = try page.alloc(f64, 3);
    defer page.free(u);

    const v: []f64 = try page.alloc(f64, 3);
    defer page.free(v);

    inline for (.{ 1.0, 5.0, 3.0 }, u) |val, *ptr| ptr.* = val;
    inline for (.{ 2.0, 3.0, 1.0 }, v) |val, *ptr| ptr.* = val;

    try hess.update(u, v);

    const A: [3][3]f64 = .{ // answers
        .{ 0x1.8a85c24f70658p+03, 0x1.44715e1c46896p+04, 0x1.be6ef01685ec3p+03 },
        .{ -0x1.0000000000000p-52, 0x1.4e2ba31c14a89p+01, 0x1.28c0f1b618468p+02 },
        .{ 0x0.0000000000000p+00, 0x0.0000000000000p+00, -0x1.dd36445718509p-01 },
    };

    for (A, hess.matrix) |A_i, R_i| {
        try testing.expect(mem.eql(f64, &A_i, R_i));
    }
}

test "(RᵀR)⋅x = b" {
    const page = std.testing.allocator;

    var hess: Hess = .{};
    try hess.init(page, 3);
    defer hess.deinit(page);

    inline for (.{ 2.0, 6.0, 8.0 }, hess.matrix[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, hess.matrix[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, hess.matrix[2]) |val, *ptr| ptr.* = val;

    const b: []f64 = try page.alloc(f64, 3);
    defer page.free(b);

    inline for (.{ 1.0, 5.0, 3.0 }, b) |val, *ptr| ptr.* = val;
    const x: []f64 = try page.alloc(f64, 3);
    defer page.free(x);

    hess.solve(x, b);

    const y: [3]f64 = .{
        -0x1.331c71c71c71cp+4,
        0x1.038e38e38e38ep+3,
        -0x1.38e38e38e38e3p+0,
    };

    try testing.expect(mem.eql(f64, &y, x));
}

fn copyto(src: []f64, des: []f64) void {
    const n: usize = src.len;
    if (n != des.len) unreachable;

    for (src, des) |src_i, *des_i| {
        des_i.* = src_i;
    }

    return;
}

fn dot(x: []f64, y: []f64) f64 {
    const n: usize = x.len;
    if (n != y.len) unreachable;

    var t: f64 = 0.0;

    if (std.simd.suggestVectorLength(f64)) |s| {
        switch (s) {
            1 => {
                for (x, y) |x_i, y_i| t += x_i * y_i;
            },
            else => {
                if (n < s) {
                    for (x, y) |x_i, y_i| t += x_i * y_i;
                } else {
                    var m: usize = @mod(n, s);

                    if (m != 0) {
                        for (x[0..m], y[0..m]) |x_i, y_i| {
                            t += x_i * y_i;
                        }
                    }

                    while (m < n) : (m += s) {
                        const a: @Vector(s, f64) = x[m..][0..s].*;
                        const b: @Vector(s, f64) = y[m..][0..s].*;
                        t += @reduce(.Add, a * b);
                    }
                }
            },
        }
    } else {
        for (x, y) |x_i, y_i| t += x_i * y_i;
    }

    return t;
}

fn ellipse_deriv(x: []f64, g: []f64) void {
    g[0] = 4 * pow3(f64, x[0] - 2) + 2 * (x[0] - 2) * pow2(f64, x[1]);
    g[1] = 2 * pow2(f64, x[0] - 2) * x[1] + 2 * (x[1] + 1);
    return;
}

fn rosenbrock_deriv(x: []f64, g: []f64) void {
    g[0] = -400 * x[0] * (x[1] - pow2(f64, x[0])) - 2 * (1 - x[0]);
    g[1] = 200 * (x[1] - pow2(f64, x[0]));
    return;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;
const testing = std.testing;

const poly = @import("./poly.zig");
const pow2 = poly.pow2;
const pow3 = poly.pow3;
