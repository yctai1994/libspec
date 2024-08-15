//! References
//! [1] J. E. Dennis, R. B. Schnabel,
//!     "Numerical Methods for Unconstrained Optimization and Nonlinear Equations,"
//!     1993, Sec. 3.4
//! [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery,
//!     "Numerical Recipes 3rd Edition: The Art of Scientific Computing,"
//!     2007, Sec. 2.10.1
//! [3] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Algorithm 3.5, 3.6
//! [4] J. Nocedal, S. J. Wright,
//!     "Numerical Optimization 2nd Edition,"
//!     2006, Procedure 18.2

xk: []f64, // xk
gk: []f64, // ∇f(xk)
pk: []f64,

xn: []f64, // xn
gn: []f64, // ∇f(xn)

sk: []f64,
yk: []f64,
ak: []f64,

uk: []f64,
vk: []f64,
bf: []f64, // buffer

Bk: [][]f64, // current approximation of Hessian

const Errors = error{
    CurvatureConditionError,
    DimensionMismatch,
    SingularError,
    SearchError,
    ZoomError,
} || mem.Allocator.Error;

const Self: type = @This();

pub fn init(allocator: mem.Allocator, n: usize) Errors!*Self {
    const ArrF64 = Array(f64){ .allocator = allocator };

    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.xk = try ArrF64.vector(n);
    errdefer allocator.free(self.xk);

    self.gk = try ArrF64.vector(n);
    errdefer allocator.free(self.gk);

    self.pk = try ArrF64.vector(n);
    errdefer allocator.free(self.pk);

    self.xn = try ArrF64.vector(n);
    errdefer allocator.free(self.xn);

    self.gn = try ArrF64.vector(n);
    errdefer allocator.free(self.gn);

    self.sk = try ArrF64.vector(n);
    errdefer allocator.free(self.sk);

    self.yk = try ArrF64.vector(n);
    errdefer allocator.free(self.yk);

    self.ak = try ArrF64.vector(n);
    errdefer allocator.free(self.ak);

    self.uk = try ArrF64.vector(n);
    errdefer allocator.free(self.uk);

    self.vk = try ArrF64.vector(n);
    errdefer allocator.free(self.vk);

    self.bf = try ArrF64.vector(n);
    errdefer allocator.free(self.bf);

    self.Bk = try ArrF64.matrix(n, n);

    for (self.Bk, 0..) |Bk_i, i| {
        @memset(Bk_i, 0.0);
        Bk_i[i] = 1.0;
    }

    return self;
}

pub fn deinit(self: *const Self, allocator: mem.Allocator) void {
    const ArrF64 = Array(f64){ .allocator = allocator };

    ArrF64.free(self.Bk);

    ArrF64.free(self.bf);
    ArrF64.free(self.vk);
    ArrF64.free(self.uk);
    ArrF64.free(self.ak);
    ArrF64.free(self.yk);
    ArrF64.free(self.sk);

    ArrF64.free(self.gn);
    ArrF64.free(self.xn);

    ArrF64.free(self.pk);
    ArrF64.free(self.gk);
    ArrF64.free(self.xk);

    allocator.destroy(self);
}

const Options = struct {
    SHOW: bool = false,
    MMAX: comptime_int = 10, // max. memorized iterations
    KMAX: comptime_int = 50, // max. iterations
    XTOL: comptime_float = 1e-16,
    GTOL: comptime_float = 1e-16,
    FTOL: comptime_float = 1e-16,

    LSC1: comptime_float = 1e-4, // line search factor 1
    LSC2: comptime_float = 9e-1, // line search factor 2
    SMAX: comptime_float = 65536.0, // line search max. step
    SMIN: comptime_float = 1e-2, // line search min. step
};

fn search(self: *const Self, obj: anytype, comptime opt: Options) Errors!void {
    const f0: f64 = obj.func(self.xk); // ϕ(0) = f(xk)
    const g0: f64 = try dot(self.pk, self.gk); // ϕ'(0) = pkᵀ⋅∇f(xk)

    if (0.0 < g0) return Errors.SearchError;

    var f_old: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var a_old: f64 = 0.0;
    var a_now: f64 = opt.SMIN;

    var iter: usize = 0;

    while (a_now < opt.SMAX) : (iter += 1) {
        for (self.xn, self.xk, self.pk) |*xn_i, xk_i, pm_i| xn_i.* = xk_i + a_now * pm_i; // xt ← xk + α⋅pk
        f_now = obj.func(self.xn); // ϕ(α) = f(xk + α⋅pk)

        // Test Wolfe conditions
        if ((f_now > f0 + opt.LSC1 * a_now * g0) or (iter > 0 and f_now > f_old)) {
            return try self.zoom(a_old, a_now, f0, g0, obj, opt);
        }

        obj.grad(self.xn, self.gn); // ∇f(xk + α⋅pk)
        g_now = try dot(self.pk, self.gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

        if (@abs(g_now) <= -opt.LSC2 * g0) break; // return a_now
        if (0.0 <= g_now) return try self.zoom(a_now, a_old, f0, g0, obj, opt);

        a_old = a_now;
        f_old = f_now;
        a_now = 2.0 * a_now;
    } else return Errors.SearchError;
}

fn zoom(self: *const Self, a_lb: f64, a_rb: f64, f0: f64, g0: f64, obj: anytype, comptime opt: Options) Errors!void {
    var f_lo: f64 = undefined;
    var f_hi: f64 = undefined;

    var g_lo: f64 = undefined;
    var g_hi: f64 = undefined;

    var a_lo: f64 = a_lb;
    var a_hi: f64 = a_rb;

    var a_now: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var iter: usize = 0;

    for (self.xn, self.xk, self.pk) |*x_lo, xk_i, sm_i| x_lo.* = xk_i + a_lo * sm_i; // xt ← xk + α_lo⋅pk
    obj.grad(self.xn, self.gn); // ∇f(xk + α_lo⋅pk)

    f_lo = obj.func(self.xn); // ϕ(α_lo) = f(xk + α_lo⋅pk)
    g_lo = try dot(self.pk, self.gn); // ϕ'(α_lo) = pkᵀ⋅∇f(xk + α_lo⋅pk)

    for (self.xn, self.xk, self.pk) |*x_hi, xk_i, sm_i| x_hi.* = xk_i + a_hi * sm_i; // xt ← xk + α_hi⋅pk
    obj.grad(self.xn, self.gn); // ∇f(xk + α_hi⋅pk)

    f_hi = obj.func(self.xn); // ϕ(α_hi) = f(xk + α_hi⋅pk)
    g_hi = try dot(self.pk, self.gn); // ϕ'(α_hi) = pkᵀ⋅∇f(xk + α_hi⋅pk), ϕ'(α_hi) can be positive

    while (iter < 10) : (iter += 1) {
        // Interpolate α
        a_now = if (a_lo < a_hi)
            try interpolate(a_lo, a_hi, f_lo, f_hi, g_lo, g_hi)
        else
            try interpolate(a_hi, a_lo, f_hi, f_lo, g_hi, g_lo);

        for (self.xn, self.xk, self.pk) |*xn_i, xk_i, sm_i| xn_i.* = xk_i + a_now * sm_i; // xt ← xk + α⋅pk
        obj.grad(self.xn, self.gn); // ∇f(xk + α⋅pk)
        f_now = obj.func(self.xn); // ϕ(α) = f(xk + α⋅pk)
        g_now = try dot(self.pk, self.gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

        if ((f_now > f0 + opt.LSC1 * a_now * g0) or (f_now > f_lo)) {
            a_hi = a_now;
            f_hi = f_now;
            g_hi = g_now;
        } else {
            if (@abs(g_now) <= -opt.LSC2 * g0) break; // return a_now
            if (0.0 <= g_now * (a_hi - a_lo)) {
                a_hi = a_lo;
                f_hi = f_lo;
                g_hi = g_lo;
            }

            a_lo = a_now;
            f_lo = f_now;
            g_lo = g_now;
        }
    }
}

fn interpolate(a_old: f64, a_new: f64, f_old: f64, f_new: f64, g_old: f64, g_new: f64) Errors!f64 {
    if (a_new <= a_old) return Errors.ZoomError;
    const d1: f64 = g_old + g_new - 3.0 * (f_old - f_new) / (a_old - a_new);
    const d2: f64 = @sqrt(d1 * d1 - g_old * g_new);
    const nu: f64 = g_new + d2 - d1;
    const de: f64 = g_new - g_old + 2.0 * d2;
    return a_new - (a_new - a_old) * (nu / de);
}

fn solve(self: *const Self, obj: anytype, comptime opt: Options) Errors!void {
    var rk: f64 = undefined; // γ = sᵀ⋅y / yᵀ⋅y
    var sn: f64 = undefined; // ‖sk‖₂, 2-norm of sk
    var ym: f64 = undefined; // max(|ykᵢ|), max-norm of yk

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    obj.grad(self.xk, self.gk); // g₀ ← ∇f(x₀)

    for (0..opt.KMAX) |kx| {
        // pk ← Bk⁻¹⋅∇f(xk)
        @memcpy(self.pk, self.gk);
        try trsv('R', 'T', self.Bk, self.pk);
        try trsv('R', 'N', self.Bk, self.pk);
        for (self.pk) |*p| p.* = -p.*; // pk ← -Bk⁻¹⋅∇f(xk)

        try self.search(obj, opt); // xn ← xk + α⋅pk, gn ← ∇f(xn)

        if (opt.SHOW) {
            debug.print("x{d: <3} = {d: >9.6}, f{d: <3} = {e: <12.10}\n", .{ kx, self.xk, kx, obj.func(self.xk) });
            debug.print("x{d: <3} = {d: >9.6}, f{d: <3} = {e: <12.10}\n", .{ kx + 1, self.xn, kx + 1, obj.func(self.xn) });
        }

        // sk ← xn - xk
        for (self.sk, self.xn, self.xk) |*sk_i, xn_i, xk_i| sk_i.* = xn_i - xk_i;
        sn = 0.0;
        for (self.sk) |sk_i| sn += pow2(sk_i);
        sn = @sqrt(sn);
        if (sn <= opt.XTOL) break;

        // yk ← ∇f(xn) - ∇f(xk) = gn - gk
        for (self.yk, self.gn, self.gk) |*yk_i, gn_i, gk_i| yk_i.* = gn_i - gk_i;
        ym = 0.0;
        for (self.yk) |ym_i| ym = @max(ym, @abs(ym_i));
        if (ym <= opt.GTOL) break;

        // secant_norm2 ← skᵀ⋅yk
        const secant_norm2: f64 = try dot(self.sk, self.yk);
        if (secant_norm2 <= 0.0) return Errors.CurvatureConditionError;

        rk = secant_norm2 / try dot(self.yk, self.yk); // γ = sᵀ⋅y / yᵀ⋅y
        if (2.2e-16 < rk) {
            // sk ← Lkᵀ⋅sk = Rk⋅sk
            trmv(self.Bk, self.sk);

            // skᵀ⋅(Lk⋅Lkᵀ)⋅sk ← skᵀ⋅sk
            const quadratic_form: f64 = try dot(self.sk, self.sk);

            // αk ← √(secant_norm2 / quadratic_form)
            const alpha_k: f64 = @sqrt(secant_norm2 / quadratic_form);

            // ak ← αk⋅Lkᵀ⋅sk = αk⋅Rk⋅sk
            for (self.ak, self.sk) |*ak_i, sk_i| ak_i.* = alpha_k * sk_i;

            // ‖ak‖ ← √(skᵀ⋅yk)
            const secant_norm: f64 = @sqrt(secant_norm2);

            // uk ← ak / ‖ak‖
            for (self.uk, self.ak) |*uk_i, ak_i| uk_i.* = ak_i / secant_norm;

            // vk ← Lk⋅ak = Rkᵀ⋅ak
            for (self.vk, 0..) |*vk_i, i| {
                vk_i.* = 0.0;
                for (0..i + 1) |j| vk_i.* += self.Bk[j][i] * self.ak[j];
            }

            // vk ← (yk - Rkᵀ⋅ak) / ‖ak‖
            for (self.vk, self.yk) |*vk_i, yk_i| vk_i.* = (yk_i - vk_i.*) / secant_norm;

            try update(self.Bk, self.uk, self.vk, self.bf);
        }

        @memcpy(self.xk, self.xn);
        @memcpy(self.gk, self.gn);
    }

    @memcpy(self.xk, self.xn);
    @memcpy(self.gk, self.gn);
}

//
// Unit-testing on Rosenbrock's functions
//

test "BFGS-B Test Case: Rosenbrock's function 2D ~ 5D" {
    const SHOW_ITERATIONS: bool = false;

    const page = testing.allocator;

    for (2..6) |n| {
        debug.print("\x1b[32m[[ Line Search Test Case: Rosenbrock's function {d}D ]]\x1b[0m\n", .{n});

        const bfgsb: *Self = try Self.init(page, n);
        defer bfgsb.deinit(page);

        const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
        for (bfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;

        try bfgsb.solve(rosenbrock, .{ .KMAX = 800, .SHOW = SHOW_ITERATIONS });
        for (bfgsb.xk, 0..) |x, i| {
            if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
        }
    }
}

fn trmv(R: [][]f64, x: []f64) void {
    var temp: f64 = undefined;
    for (R, x, 0..) |R_i, *x_i, i| {
        temp = 0.0;
        for (R_i[i..], x[i..]) |R_ij, x_j| {
            temp += R_ij * x_j;
        }
        x_i.* = temp;
    }
}

fn update(R: [][]f64, u: []f64, v: []f64, b: []f64) Errors!void {
    const n: usize = u.len;
    if (n != v.len) unreachable;

    // Find largest k such that u[k] ≠ 0.
    var k: usize = 0;
    for (b, u, 0..) |*ptr, val, index| {
        if (val != 0.0) k = index;
        ptr.* = val;
    }

    // Transform R + u⋅vᵀ to upper Hessenberg.
    if (0 < k) {
        var i: usize = k - 1;
        while (0 <= i) : (i -= 1) {
            rotate(R, i, n, b[i], -b[i + 1]);

            if (b[i] == 0.0) {
                b[i] = @abs(b[i + 1]);
            } else if (@abs(b[i]) > @abs(b[i + 1])) {
                b[i] = @abs(b[i]) * @sqrt(1.0 + pow2(b[i + 1] / b[i]));
            } else {
                b[i] = @abs(b[i + 1]) * @sqrt(1.0 + pow2(b[i] / b[i + 1]));
            }

            if (i == 0) break;
        }
    }

    for (R[0], v) |*R_0i, v_i| {
        R_0i.* += b[0] * v_i;
    }

    // Transform upper Hessenberg matrix to upper triangular.
    for (0..k, 1..) |j, jp1| {
        rotate(R, j, n, R[j][j], -R[jp1][j]);
    }

    for (R, 0..) |R_j, j| {
        if (R_j[j] == 0.0) return Errors.SingularError;
    }
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
}

test "Hess.update" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const R: [][]f64 = try ArrF64.matrix(3, 3);
    defer ArrF64.free(R);

    inline for (.{ 2.0, 6.0, 8.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, R[2]) |val, *ptr| ptr.* = val;

    const u: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(u);

    const v: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(v);

    const b: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(b);

    inline for (.{ 1.0, 5.0, 3.0 }, u) |val, *ptr| ptr.* = val;
    inline for (.{ 2.0, 3.0, 1.0 }, v) |val, *ptr| ptr.* = val;

    try update(R, u, v, b);

    const A: [3][3]f64 = .{ // answers
        .{ 0x1.8a85c24f70658p+03, 0x1.44715e1c46896p+04, 0x1.be6ef01685ec3p+03 },
        .{ -0x1.0000000000000p-52, 0x1.4e2ba31c14a89p+01, 0x1.28c0f1b618468p+02 },
        .{ 0x0.0000000000000p+00, 0x0.0000000000000p+00, -0x1.dd36445718509p-01 },
    };

    for (A, R) |A_i, R_i| {
        try testing.expect(mem.eql(f64, &A_i, R_i));
    }
}

test "(RᵀR)⋅x = b" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const R: [][]f64 = try ArrF64.matrix(3, 3);
    defer ArrF64.free(R);

    inline for (.{ 2.0, 6.0, 8.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 3.0 }, R[2]) |val, *ptr| ptr.* = val;

    const x: []f64 = try ArrF64.vector(3);
    defer ArrF64.free(x);

    inline for (.{ 1.0, 5.0, 3.0 }, x) |val, *ptr| ptr.* = val;

    try trsv('R', 'T', R, x);
    try trsv('R', 'N', R, x);

    const y: [3]f64 = .{
        -0x1.331c71c71c71cp+4,
        0x1.038e38e38e38ep+3,
        -0x1.38e38e38e38e3p+0,
    };

    try testing.expect(mem.eql(f64, &y, x));
}

//
// Subroutines
//

fn Array(comptime T: type) type {
    // already comptime scope
    const slice_al: comptime_int = @alignOf([]T);
    const child_al: comptime_int = @alignOf(T);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(T);

    return struct {
        allocator: std.mem.Allocator,

        fn matrix(self: @This(), nrow: usize, ncol: usize) Errors![][]T {
            const buff: []u8 = try self.allocator.alloc(u8, nrow * ncol * child_sz + nrow * slice_sz);

            const mat: [][]T = blk: {
                const ptr: [*]align(slice_al) []T = @ptrCast(@alignCast(buff.ptr));
                break :blk ptr[0..nrow];
            };

            const chunk_sz: usize = ncol * child_sz;
            var padding: usize = nrow * slice_sz;

            for (mat) |*row| {
                row.* = blk: {
                    const ptr: [*]align(child_al) T = @ptrCast(@alignCast(buff.ptr + padding));
                    break :blk ptr[0..ncol];
                };
                padding += chunk_sz;
            }

            return mat;
        }

        fn vector(self: @This(), n: usize) Errors![]T {
            return try self.allocator.alloc(T, n);
        }

        fn free(self: @This(), slice: anytype) void {
            const S: type = comptime @TypeOf(slice);

            switch (S) {
                [][]T => {
                    const ptr: [*]u8 = @ptrCast(@alignCast(slice.ptr));
                    const len: usize = blk: {
                        const nrow: usize = slice.len;
                        const ncol: usize = slice[0].len;
                        break :blk nrow * ncol * child_sz + nrow * slice_sz;
                    };

                    self.allocator.free(ptr[0..len]);
                },
                []T => {
                    self.allocator.free(slice);
                },
                else => @compileError("Invalid type: " ++ @typeName(T)),
            }

            return;
        }
    };
}

fn dot(x: []f64, y: []f64) Errors!f64 {
    const n: usize = x.len;
    if (n != y.len) return Errors.DimensionMismatch;

    var t: f64 = 0.0;
    for (0..n) |i| t += x[i] * y[i];

    return t;
}

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const math = std.math;
const debug = std.debug;
const testing = std.testing;

const trsv = @import("./trsv.zig").trsv;
const Rosenbrock = @import("./Rosenbrock.zig");
