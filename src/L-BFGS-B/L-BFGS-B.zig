//! References
//! [1] J. Nocedal, S. J. Wright, 2006, "Numerical Optimization 2nd Edition"

const Options = struct {
    MMAX: comptime_int = 10, // max. memorized iterations
    KMAX: comptime_int = 50, // max. iterations
    XTOL: comptime_float = 1e-16,
    GTOL: comptime_float = 1e-16,
    FTOL: comptime_float = 1e-16,

    LSC1: comptime_float = 1e-4, // line search factor 1
    LSC2: comptime_float = 9e-1, // line search factor 2
    SMAX: comptime_float = 65536.0, // line search max. step
    SMIN: comptime_float = 1e-1, // line search min. step
};

const Errors = error{
    SearchError,
    ZoomError,
} || mem.Allocator.Error;

fn LBFGSB(comptime opt: Options) type {
    if (opt.MMAX < 1) @compileError("Options.MMAX should be ≥ 1");
    if (opt.KMAX < 1) @compileError("Options.KMAX should be ≥ 1");

    return struct {
        xk: []f64,
        gk: []f64,
        pk: []f64,
        uk: []f64,
        vk: []f64,

        xn: []f64,
        gn: []f64,

        Sk: [][]f64,
        Yk: [][]f64,

        Skgk: []f64,
        Ykgk: []f64,

        YkSk: [][]f64,
        YkYk: [][]f64,

        const Self: type = @This();

        fn init(allocator: mem.Allocator, n: usize) Errors!*Self {
            const ArrF64 = Array(f64){ .allocator = allocator };

            const self: *Self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.xk = try ArrF64.vector(n);
            errdefer ArrF64.free(self.xk);

            self.gk = try ArrF64.vector(n);
            errdefer ArrF64.free(self.gk);

            self.pk = try ArrF64.vector(n);
            errdefer ArrF64.free(self.pk);

            self.uk = try ArrF64.vector(opt.MMAX);
            errdefer ArrF64.free(self.uk);

            self.vk = try ArrF64.vector(opt.MMAX);
            errdefer ArrF64.free(self.vk);

            self.xn = try ArrF64.vector(n);
            errdefer ArrF64.free(self.xn);

            self.gn = try ArrF64.vector(n);
            errdefer ArrF64.free(self.gn);

            self.Sk = try ArrF64.matrix(opt.MMAX, n);
            errdefer ArrF64.free(self.Sk);

            self.Yk = try ArrF64.matrix(opt.MMAX, n);
            errdefer ArrF64.free(self.Yk);

            self.Skgk = try ArrF64.vector(opt.MMAX);
            errdefer ArrF64.free(self.Skgk);

            self.Ykgk = try ArrF64.vector(opt.MMAX);
            errdefer ArrF64.free(self.Ykgk);

            self.YkSk = try ArrF64.matrix(opt.MMAX, opt.MMAX);
            errdefer ArrF64.free(self.YkSk);

            self.YkYk = try ArrF64.matrix(opt.MMAX, opt.MMAX);

            return self;
        }

        fn deinit(self: *const Self, allocator: mem.Allocator) void {
            const ArrF64 = Array(f64){ .allocator = allocator };

            ArrF64.free(self.YkYk);
            ArrF64.free(self.YkSk);

            ArrF64.free(self.Ykgk);
            ArrF64.free(self.Skgk);

            ArrF64.free(self.Yk);
            ArrF64.free(self.Sk);

            ArrF64.free(self.gn);
            ArrF64.free(self.xn);

            ArrF64.free(self.vk);
            ArrF64.free(self.uk);
            ArrF64.free(self.pk);
            ArrF64.free(self.gk);
            ArrF64.free(self.xk);

            allocator.destroy(self);
        }

        // Algorithm 3.5 in [1]
        fn search(self: *const Self, obj: anytype) Errors!void {
            const f0: f64 = obj.func(self.xk); // ϕ(0) = f(xk)
            const g0: f64 = dot(self.pk, self.gk); // ϕ'(0) = pkᵀ⋅∇f(xk)

            if (0.0 < g0) return Errors.SearchError;

            var f_old: f64 = undefined;
            var f_now: f64 = undefined;
            var g_now: f64 = undefined;

            var a_old: f64 = 0.0;
            var a_now: f64 = opt.SMIN;

            var iter: usize = 0;

            while (a_now < opt.SMAX) : (iter += 1) {
                for (self.xn, self.xk, self.pk) |*xn_i, xm_i, pm_i| xn_i.* = xm_i + a_now * pm_i; // xt ← xk + α⋅pk
                f_now = obj.func(self.xn); // ϕ(α) = f(xk + α⋅pk)

                // Test Wolfe conditions
                if ((f_now > f0 + opt.LSC1 * a_now * g0) or (iter > 0 and f_now > f_old)) {
                    return try self.zoom(a_old, a_now, f0, g0, obj);
                }

                obj.grad(self.xn, self.gn); // ∇f(xk + α⋅pk)
                g_now = dot(self.pk, self.gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

                if (@abs(g_now) <= -opt.LSC2 * g0) break; // return a_now
                if (0.0 <= g_now) return try self.zoom(a_now, a_old, f0, g0, obj);

                a_old = a_now;
                f_old = f_now;
                a_now = 2.0 * a_now;
            } else return Errors.SearchError;
        }

        // Algorithm 3.6 in [1]
        fn zoom(self: *const Self, a_lb: f64, a_rb: f64, f0: f64, g0: f64, obj: anytype) Errors!void {
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

            for (self.xn, self.xk, self.pk) |*x_lo, xm_i, sm_i| x_lo.* = xm_i + a_lo * sm_i; // xt ← xk + α_lo⋅pk
            obj.grad(self.xn, self.gn); // ∇f(xk + α_lo⋅pk)

            f_lo = obj.func(self.xn); // ϕ(α_lo) = f(xk + α_lo⋅pk)
            g_lo = dot(self.pk, self.gn); // ϕ'(α_lo) = pkᵀ⋅∇f(xk + α_lo⋅pk)

            for (self.xn, self.xk, self.pk) |*x_hi, xm_i, sm_i| x_hi.* = xm_i + a_hi * sm_i; // xt ← xk + α_hi⋅pk
            obj.grad(self.xn, self.gn); // ∇f(xk + α_hi⋅pk)

            f_hi = obj.func(self.xn); // ϕ(α_hi) = f(xk + α_hi⋅pk)
            g_hi = dot(self.pk, self.gn); // ϕ'(α_hi) = pkᵀ⋅∇f(xk + α_hi⋅pk), ϕ'(α_hi) can be positive

            while (iter < 10) : (iter += 1) {
                // Interpolate α
                a_now = if (a_lo < a_hi)
                    try interpolate(a_lo, a_hi, f_lo, f_hi, g_lo, g_hi)
                else
                    try interpolate(a_hi, a_lo, f_hi, f_lo, g_hi, g_lo);

                for (self.xn, self.xk, self.pk) |*xn_i, xm_i, sm_i| xn_i.* = xm_i + a_now * sm_i; // xt ← xk + α⋅pk
                obj.grad(self.xn, self.gn); // ∇f(xk + α⋅pk)
                f_now = obj.func(self.xn); // ϕ(α) = f(xk + α⋅pk)
                g_now = dot(self.pk, self.gn); // ϕ'(α) = pkᵀ⋅∇f(xk + α⋅pk)

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

        // Equation 3.59 in [1]
        fn interpolate(a_old: f64, a_new: f64, f_old: f64, f_new: f64, g_old: f64, g_new: f64) Errors!f64 {
            if (a_new <= a_old) return Errors.ZoomError;
            const d1: f64 = g_old + g_new - 3.0 * (f_old - f_new) / (a_old - a_new);
            const d2: f64 = @sqrt(d1 * d1 - g_old * g_new);
            const nu: f64 = g_new + d2 - d1;
            const de: f64 = g_new - g_old + 2.0 * d2;
            return a_new - (a_new - a_old) * (nu / de);
        }

        fn solve(self: *const Self, obj: anytype) Errors!void {
            var ix: usize = undefined;
            var jx: usize = undefined;
            var kx: usize = undefined;
            var rk: f64 = undefined; // γ = sᵀ⋅y / yᵀ⋅y
            var sn: f64 = undefined; // ‖sk‖₂, 2-norm of sk
            var ym: f64 = undefined; // max(|ykᵢ|), max-norm of yk
            var tk: f64 = undefined;

            // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

            kx = 0;

            obj.grad(self.xk, self.gk); // g₀ ← ∇f(x₀)
            for (self.pk, self.gk) |*p, g| p.* = -g; // p₀ ← p₀ = -∇f(x₀)

            try self.search(obj); // x₁ ← x₀ + α⋅p₀, g₁ ← ∇f(x₁)

            debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx, self.xk, kx, obj.func(self.xk) });
            debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx + 1, self.xn, kx + 1, obj.func(self.xn) });

            // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

            kx += 1;

            while (kx < opt.MMAX) : (kx += 1) {
                ix = kx - 1;

                while (0 < ix) : (ix -= 1) {
                    jx = ix - 1;
                    @memcpy(self.YkSk[ix][1..kx], self.YkSk[jx][0 .. kx - 1]);
                    @memcpy(self.YkYk[ix][1..kx], self.YkYk[jx][0 .. kx - 1]);

                    @memcpy(self.Sk[ix], self.Sk[jx]);
                    @memcpy(self.Yk[ix], self.Yk[jx]);
                }

                for (self.Sk[0], self.xn, self.xk) |*sk_i, xn_i, xk_i| sk_i.* = xn_i - xk_i;

                sn = 0.0;
                for (self.Sk[0]) |sk_i| sn += pow2(sk_i);
                sn = @sqrt(sn);
                if (sn <= opt.XTOL) return;

                for (self.Yk[0], self.gn, self.gk) |*yk_i, gn_i, gk_i| yk_i.* = gn_i - gk_i;

                ym = 0.0;
                for (self.Yk[0]) |yk_i| ym = @max(ym, @abs(yk_i));
                if (ym <= opt.GTOL) return;

                for (1..kx) |i| {
                    self.YkSk[0][i] = dot(self.Yk[0], self.Sk[i]);
                    self.YkSk[i][0] = dot(self.Yk[i], self.Sk[0]);

                    self.YkYk[0][i] = dot(self.Yk[0], self.Yk[i]);
                    self.YkYk[i][0] = self.YkYk[0][i];
                }

                self.YkYk[0][0] = dot(self.Yk[0], self.Yk[0]);
                self.YkSk[0][0] = dot(self.Yk[0], self.Sk[0]);

                debug.print("Sk   =\n", .{});
                for (self.Sk[0..kx]) |S_i| debug.print("  {e: >12.10}\n", .{S_i});
                debug.print("Yk   =\n", .{});
                for (self.Yk[0..kx]) |Y_i| debug.print("  {e: >12.10}\n", .{Y_i});
                debug.print("YkSk =\n", .{});
                for (self.YkSk[0..kx]) |YS_i| debug.print("  {e: >12.10}\n", .{YS_i[0..kx]});
                debug.print("YkYk =\n", .{});
                for (self.YkYk[0..kx]) |YY_i| debug.print("  {e: >12.10}\n", .{YY_i[0..kx]});

                // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

                @memcpy(self.xk, self.xn);
                @memcpy(self.gk, self.gn);

                rk = self.YkSk[0][0] / self.YkYk[0][0]; // γ = sᵀ⋅y / yᵀ⋅y
                if (rk <= 0.0) return Errors.ZoomError;

                for (self.Skgk[0..kx], self.Sk[0..kx]) |*Sg_i, s| Sg_i.* = dot(s, self.gk);
                for (self.Ykgk[0..kx], self.Yk[0..kx]) |*Yg_i, y| Yg_i.* = dot(y, self.gk);

                // Rk⁻ᵀ⋅Skgk

                @memcpy(self.vk[0..kx], self.Skgk[0..kx]);

                for (self.YkSk[0..kx], self.vk[0..kx], 0.., 1..) |YS_i, *v_i, i, ip1| {
                    v_i.* /= YS_i[i];
                    for (YS_i[ip1..kx], self.vk[ip1..kx]) |YS_ij, *v_j| v_j.* -= YS_ij * v_i.*;
                }

                for (self.YkSk[0..kx], self.YkYk[0..kx], self.Ykgk[0..kx], 0..) |YS_j, YY_j, Yg_j, j| {
                    self.uk[j] = YS_j[j] * self.vk[j] + rk * dot(YY_j[0..kx], self.vk[0..kx]) - rk * Yg_j;
                }

                for (self.vk[0..kx]) |*v_i| v_i.* = -v_i.*;

                // Rk⁻¹⋅(...)

                ix = kx - 1;

                while (true) : (ix -= 1) {
                    tk = self.uk[ix];
                    jx = ix + 1;
                    for (self.YkSk[ix][jx..kx], self.uk[jx..kx]) |YS_ij, u_j| tk -= YS_ij * u_j;
                    self.uk[ix] = tk / self.YkSk[ix][ix];
                    if (ix == 0) break;
                }

                for (self.pk, self.gk) |*p, g| p.* = -rk * g;
                for (self.uk[0..kx], self.vk[0..kx], self.Sk[0..kx], self.Yk[0..kx]) |u_i, v_i, S_i, Y_i| {
                    for (self.pk, S_i, Y_i) |*p, S_ij, Y_ij| p.* -= u_i * S_ij + rk * v_i * Y_ij;
                }

                debug.print("Skgk = {e: >12.10}\nYkgk = {e: >12.10}\n", .{ self.Skgk[0..kx], self.Ykgk[0..kx] });
                debug.print("wk1  = {e: >12.10}\n", .{self.uk[0..kx]});
                debug.print("wk2  = {e: >12.10}\n", .{self.vk[0..kx]});
                debug.print("pk   = {e: >12.10}\n", .{self.pk});

                try self.search(obj); // xn ← xk + α⋅pk, gn ← ∇f(xn)

                debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx, self.xk, kx, obj.func(self.xk) });
                debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx + 1, self.xn, kx + 1, obj.func(self.xn) });
            }

            while (kx <= opt.KMAX) : (kx += 1) {
                ix = comptime opt.MMAX - 1;

                while (0 < ix) : (ix -= 1) {
                    jx = ix - 1;

                    @memcpy(self.YkSk[ix][1..opt.MMAX], self.YkSk[jx][0 .. opt.MMAX - 1]);
                    @memcpy(self.YkYk[ix][1..opt.MMAX], self.YkYk[jx][0 .. opt.MMAX - 1]);

                    @memcpy(self.Sk[ix], self.Sk[jx]);
                    @memcpy(self.Yk[ix], self.Yk[jx]);
                }

                for (self.Sk[0], self.xn, self.xk) |*sk_i, xn_i, xk_i| sk_i.* = xn_i - xk_i;

                sn = 0.0;
                for (self.Sk[0]) |sk_i| sn += pow2(sk_i);
                sn = @sqrt(sn);
                if (sn <= opt.XTOL) return;

                for (self.Yk[0], self.gn, self.gk) |*yk_i, gn_i, gk_i| yk_i.* = gn_i - gk_i;

                ym = 0.0;
                for (self.Yk[0]) |yk_i| ym = @max(ym, @abs(yk_i));
                if (ym <= opt.GTOL) return;

                for (1..opt.MMAX) |i| {
                    self.YkSk[0][i] = dot(self.Yk[0], self.Sk[i]);
                    self.YkSk[i][0] = dot(self.Yk[i], self.Sk[0]);

                    self.YkYk[0][i] = dot(self.Yk[0], self.Yk[i]);
                    self.YkYk[i][0] = self.YkYk[0][i];
                }

                self.YkYk[0][0] = dot(self.Yk[0], self.Yk[0]);
                self.YkSk[0][0] = dot(self.Yk[0], self.Sk[0]);

                debug.print("Sk   =\n", .{});
                for (self.Sk) |S_i| debug.print("  {e: >12.10}\n", .{S_i});
                debug.print("Yk   =\n", .{});
                for (self.Yk) |Y_i| debug.print("  {e: >12.10}\n", .{Y_i});
                debug.print("YkSk =\n", .{});
                for (self.YkSk) |YS_i| debug.print("  {e: >12.10}\n", .{YS_i});
                debug.print("YkYk =\n", .{});
                for (self.YkYk) |YY_i| debug.print("  {e: >12.10}\n", .{YY_i});

                // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

                @memcpy(self.xk, self.xn);
                @memcpy(self.gk, self.gn);

                rk = self.YkSk[0][0] / self.YkYk[0][0]; // γ = sᵀ⋅y / yᵀ⋅y
                if (rk <= 0.0) return Errors.ZoomError;

                for (self.Skgk, self.Sk) |*Sg_i, s| Sg_i.* = dot(s, self.gk);
                for (self.Ykgk, self.Yk) |*Yg_i, y| Yg_i.* = dot(y, self.gk);

                // Rk⁻ᵀ⋅Skgk

                @memcpy(self.vk, self.Skgk);

                for (self.YkSk, self.vk, 0.., 1..) |YS_i, *v_i, i, ip1| {
                    v_i.* /= YS_i[i];
                    for (YS_i[ip1..], self.vk[ip1..]) |YS_ij, *v_j| v_j.* -= YS_ij * v_i.*;
                }

                for (self.uk, self.vk, self.YkSk, self.YkYk, self.Ykgk, 0..) |*u_i, v_i, YS_i, YY_i, Yg_i, i| {
                    u_i.* = YS_i[i] * v_i + rk * dot(YY_i, self.vk) - rk * Yg_i;
                }

                for (self.vk) |*v_i| v_i.* = -v_i.*;

                // Rk⁻¹⋅(...)

                ix = comptime opt.MMAX - 1;

                while (true) : (ix -= 1) {
                    tk = self.uk[ix];
                    jx = ix + 1;
                    for (self.YkSk[ix][jx..], self.uk[jx..]) |YS_ij, u_j| tk -= YS_ij * u_j;
                    self.uk[ix] = tk / self.YkSk[ix][ix];
                    if (ix == 0) break;
                }

                for (self.pk, self.gk) |*p, g| p.* = -rk * g;
                for (self.uk, self.vk, self.Sk, self.Yk) |u_i, v_i, S_i, Y_i| {
                    for (self.pk, S_i, Y_i) |*p, S_ij, Y_ij| p.* -= u_i * S_ij + rk * v_i * Y_ij;
                }

                debug.print("Skgk = {e: >12.10}\nYkgk = {e: >12.10}\n", .{ self.Skgk, self.Ykgk });
                debug.print("wk1  = {e: >12.10}\n", .{self.uk});
                debug.print("wk2  = {e: >12.10}\n", .{self.vk});
                debug.print("pk   = {e: >12.10}\n", .{self.pk});

                try self.search(obj); // xn ← xk + α⋅pk, gn ← ∇f(xn)

                debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx, self.xk, kx, obj.func(self.xk) });
                debug.print("x{d: <3} = {d: >12.2}, f{d: <3} = {e: <12.10}\n", .{ kx + 1, self.xn, kx + 1, obj.func(self.xn) });
            }
        }
    };
}

//
// Unit-testing on Rosenbrock's functions
//

test "L-BFGS-B Test Case: Rosenbrock's function 2D" {
    const n: comptime_int = 2;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{@as(u8, n)});

    const page = testing.allocator;

    const lbfgsb = try LBFGSB(.{ .KMAX = 300 }).init(page, n);
    defer lbfgsb.deinit(page);

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    for (lbfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;

    try lbfgsb.solve(rosenbrock);
    for (lbfgsb.xn, 0..) |x, i| {
        if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
    }
}

test "L-BFGS-B Test Case: Rosenbrock's function 3D" {
    const n: comptime_int = 3;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{@as(u8, n)});

    const page = testing.allocator;

    const lbfgsb = try LBFGSB(.{ .KMAX = 400 }).init(page, n);
    defer lbfgsb.deinit(page);

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    for (lbfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;

    try lbfgsb.solve(rosenbrock);
    for (lbfgsb.xn, 0..) |x, i| {
        if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
    }
}

test "L-BFGS-B Test Case: Rosenbrock's function 4D" {
    const n: comptime_int = 4;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{@as(u8, n)});

    const page = testing.allocator;

    const lbfgsb = try LBFGSB(.{ .KMAX = 300 }).init(page, n);
    defer lbfgsb.deinit(page);

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    for (lbfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;

    try lbfgsb.solve(rosenbrock);
    for (lbfgsb.xn, 0..) |x, i| {
        if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
    }
}

test "L-BFGS-B Test Case: Rosenbrock's function 5D" {
    const n: comptime_int = 5;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{@as(u8, n)});

    const page = testing.allocator;

    const lbfgsb = try LBFGSB(.{ .KMAX = 300 }).init(page, n);
    defer lbfgsb.deinit(page);

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    for (lbfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;

    try lbfgsb.solve(rosenbrock);
    for (lbfgsb.xn, 0..) |x, i| {
        if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
    }
}

const Rosenbrock = struct {
    a: f64,
    b: f64,

    fn subfunc(self: *const Rosenbrock, x: f64, y: f64) f64 {
        return pow2(self.a - x) + self.b * pow2(y - pow2(x));
    }
    fn func(self: *const Rosenbrock, x: []f64) f64 {
        const n: usize = x.len;
        var val: f64 = 0.0;
        for (0..n - 1) |i| val += self.subfunc(x[i], x[i + 1]);
        return val;
    }

    fn subgrad1(self: *const Rosenbrock, x: f64, y: f64) f64 {
        return 2.0 * (x - self.a) + 4.0 * self.b * x * (pow2(x) - y);
    }

    fn subgrad2(self: *const Rosenbrock, x: f64, y: f64) f64 {
        return 2.0 * self.b * (y - pow2(x));
    }

    fn grad(self: *const Rosenbrock, x: []f64, g: []f64) void {
        const n: usize = x.len;
        if (n != g.len) unreachable;
        const m: usize = n - 1;

        g[0] = self.subgrad1(x[0], x[1]);
        for (1..m) |i| g[i] = self.subgrad1(x[i], x[i + 1]) + self.subgrad2(x[i - 1], x[i]);
        g[m] = self.subgrad2(x[m - 1], x[m]);
    }
};

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

fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const mem = std.mem;
const debug = std.debug;
const testing = std.testing;
