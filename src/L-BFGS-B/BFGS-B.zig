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

xk: []f64, // k-th location, xk
gk: []f64, // k-th gradient, ∇f(xk)
xc: []f64, // k-th Cauchy point
pk: []f64, // k-th search direction

tk: []f64, // temp buffer, for finding Cauchy point
xt: []f64, // temp buffer, for finding Cauchy point

xn: []f64, // xn
gn: []f64, // ∇f(xn)

sk: []f64,
yk: []f64,
ak: []f64,

uk: []f64,
vk: []f64,
bf: []f64, // buffer

Bk: [][]f64, // current approximation of Hessian

AHA: [][]f64, // temp buffer, for solving quadratic subproblem
iWk: []usize, // k-th working set

const Errors = error{
    CurvatureConditionError,
    DescentDirectionError,
    NotPositiveDefinite,
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
    errdefer ArrF64.free(self.xk);

    self.gk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.gk);

    self.xc = try ArrF64.vector(n);
    errdefer ArrF64.free(self.xc);

    self.pk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.pk);

    self.tk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.tk);

    self.xt = try ArrF64.vector(n);
    errdefer ArrF64.free(self.xt);

    self.xn = try ArrF64.vector(n);
    errdefer ArrF64.free(self.xn);

    self.gn = try ArrF64.vector(n);
    errdefer ArrF64.free(self.gn);

    self.sk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.sk);

    self.yk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.yk);

    self.ak = try ArrF64.vector(n);
    errdefer ArrF64.free(self.ak);

    self.uk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.uk);

    self.vk = try ArrF64.vector(n);
    errdefer ArrF64.free(self.vk);

    self.bf = try ArrF64.vector(n);
    errdefer ArrF64.free(self.bf);

    self.Bk = try ArrF64.matrix(n, n);
    errdefer ArrF64.free(self.Bk);

    self.AHA = try ArrF64.matrix(n, n);
    errdefer ArrF64.free(self.AHA);

    self.iWk = try allocator.alloc(usize, n);
    errdefer allocator.free(self.iWk);

    for (0..n) |i| {
        @memset(self.Bk[i], 0.0);
        self.Bk[i][i] = 1.0;
    }

    return self;
}

pub fn deinit(self: *const Self, allocator: mem.Allocator) void {
    const ArrF64 = Array(f64){ .allocator = allocator };

    ArrF64.free(self.xk);
    ArrF64.free(self.gk);
    ArrF64.free(self.xc);
    ArrF64.free(self.pk);

    ArrF64.free(self.tk);
    ArrF64.free(self.xt);

    ArrF64.free(self.xn);
    ArrF64.free(self.gn);

    ArrF64.free(self.sk);
    ArrF64.free(self.yk);
    ArrF64.free(self.ak);

    ArrF64.free(self.uk);
    ArrF64.free(self.vk);
    ArrF64.free(self.bf);

    ArrF64.free(self.Bk);
    ArrF64.free(self.AHA);

    allocator.free(self.iWk);
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

fn search(self: *const Self, obj: anytype, opt_smin: ?f64, opt_smax: ?f64, comptime opt: Options) Errors!void {
    const smin: f64 = @max(opt.SMIN, opt_smin orelse opt.SMIN);
    const smax: f64 = @min(opt.SMAX, opt_smax orelse opt.SMAX);

    const f0: f64 = obj.func(self.xk); // ϕ(0) = f(xk)
    const g0: f64 = try dot(self.pk, self.gk); // ϕ'(0) = pkᵀ⋅∇f(xk)

    if (0.0 < g0) return Errors.DescentDirectionError;

    var f_old: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var a_old: f64 = 0.0;
    var a_now: f64 = smin;

    var iter: usize = 0;

    while (a_now <= smax) : (iter += 1) {
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

fn project(self: *const Self, lb: []f64, ub: []f64, ta: *usize, comptime opt: Options) Errors!void {
    const n: usize = self.xk.len;
    for (0..n) |i| {
        if (self.gk[i] < 0.0 and ub[i] < math.inf(f64)) {
            self.tk[i] = (self.xk[i] - ub[i]) / self.gk[i];
        } else if (0.0 < self.gk[i] and -math.inf(f64) < lb[i]) {
            self.tk[i] = (self.xk[i] - lb[i]) / self.gk[i];
        } else self.tk[i] = math.inf(f64);
    }

    @memcpy(self.gn, self.tk); // use gn as buffer, for sorting
    if (opt.SHOW) debug.print("  tk       = {d: >9.6}\n", .{self.tk});

    insertionSort(self.gn); // sorted tk
    if (opt.SHOW) debug.print("  sort(tk) = {d: >9.6}\n", .{self.gn});

    var t_left: f64 = 0.0;

    var dt: f64 = undefined;
    var dt_min: f64 = undefined;

    var df: f64 = undefined;
    var ddf: f64 = undefined;

    @memcpy(self.xt, self.xk); // x(t_0) and x( t(j-1) )

    for (self.gn) |t_right| { // for j = 1, 2, ...
        if (t_left == t_right) {
            if (opt.SHOW) debug.print("  t_left == t_right\n", .{});
            continue;
        }
        dt = t_right - t_left;

        for (0..n) |i| self.pk[i] = if (t_left < self.tk[i]) -self.gk[i] else 0.0; // p(j-1)

        @memcpy(self.xc, self.xt); // xc ← x( t(j-1) )
        // pᵀ⋅( Bk⋅x( t(j-1) ) + gk ) = pᵀ⋅( (Lk⋅Rk)⋅x( t(j-1) ) + gk )
        try trmv('R', 'N', self.Bk, self.xc);
        try trmv('R', 'T', self.Bk, self.xc);
        for (self.xc, self.gk) |*xc_i, gk_i| xc_i.* += gk_i;
        df = try dot(self.pk, self.xc);

        // pᵀ⋅Bk⋅p = pᵀ⋅(Lk⋅Rk)⋅p
        try trmv('R', 'N', self.Bk, self.pk);
        ddf = try dot(self.pk, self.pk);
        dt_min = -df / ddf;

        if (opt.SHOW) debug.print("  dt = {d: >9.6}, dt_min = {d: >9.6}\n", .{ dt, dt_min });

        if (dt <= dt_min) {
            if (opt.SHOW) debug.print("  << dt <= dt_min >>\n", .{});
            for (0..n) |i| self.xt[i] += if (t_left < self.tk[i]) -dt * self.gk[i] else 0.0;
            t_left = t_right;
        } else break;
    }

    dt_min = @max(0.0, dt_min);
    t_left += dt_min;

    ta.* = 0;

    for (0..n) |i| {
        if (t_left < self.tk[i]) {
            self.xc[i] = self.xk[i] - t_left * self.gk[i];
        } else {
            self.xc[i] = self.xk[i] - self.tk[i] * self.gk[i];
            self.iWk[ta.*] = i;
            ta.* += 1;
        }
    }

    if (opt.SHOW) {
        debug.print("  dt_min = {d: >9.6}, t_left = {d: >9.6}\n", .{ dt_min, t_left });
        debug.print("  Cauchy point xc = {d: >9.6}, iWk = {d} (ta = {d})\n", .{ self.xc, self.iWk[0..ta.*], ta.* });
    }
}

fn solve(self: *const Self, obj: anytype, lb: []f64, ub: []f64, comptime opt: Options) Errors!void {
    var rk: f64 = undefined; // γ = sᵀ⋅y / yᵀ⋅y
    var sn: f64 = undefined; // ‖sk‖₂, 2-norm of sk
    var ym: f64 = undefined; // max(|ykᵢ|), max-norm of yk

    var ta: usize = undefined;

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    obj.grad(self.xk, self.gk); // g₀ ← ∇f(x₀)

    if (opt.SHOW) debug.print("x{d: <3} = {d: >9.6}, f{d: <3} = {e: <12.10}\n", .{ 0, self.xk, 0, obj.func(self.xk) });

    for (0..opt.KMAX) |kx| {
        try self.project(lb, ub, &ta, opt);

        if (0 < ta) {
            debug.print("  = = = = =\n", .{});

            for (0..ta, self.iWk[0..ta]) |ia, ja| {
                @memset(self.AHA[ia], 0.0);
                self.AHA[ia][ja] = 1.0;
            }

            for (0..ta, self.iWk[0..ta]) |ia, ja| {
                debug.print("  (ia, ja) = ({d}, {d}) => {d: >9.6}\n", .{ ia, ja, self.AHA[ia] });
            }

            debug.print("  = = = = =\n", .{});

            for (0..ta) |ia| try trsv('R', 'T', self.Bk, self.AHA[ia]);
            for (0..ta) |ia| try trsv('R', 'N', self.Bk, self.AHA[ia]);

            for (0..ta, self.iWk[0..ta]) |ia, ja| {
                debug.print("  (ia, ja) = ({d}, {d}) => {d: >9.6}\n", .{ ia, ja, self.AHA[ia] });
            }

            debug.print("  = = = = =\n", .{});

            for (0..ta) |i| {
                for (0..ta, self.iWk[0..ta]) |j, ja| {
                    self.AHA[i][j] = self.AHA[i][ja];
                }
            }

            for (0..ta, self.iWk[0..ta]) |ia, ja| {
                debug.print("  (ia, ja) = ({d}, {d}) => {d: >9.6}\n", .{ ia, ja, self.AHA[ia][0..ta] });
            }

            debug.print("  = = = = =\n", .{});

            try cholesky(self.AHA[0..ta]);

            for (0..ta, self.iWk[0..ta]) |ia, ja| {
                debug.print("  (ia, ja) = ({d}, {d}) => {d: >9.6}\n", .{ ia, ja, self.AHA[ia][0..ta] });
            }

            // pk ← Bk⁻¹⋅∇f(xk)
            @memcpy(self.pk, self.gk);
            try trsv('R', 'T', self.Bk, self.pk);
            try trsv('R', 'N', self.Bk, self.pk);
            for (self.pk, self.xc, self.xk) |*pk_i, xc_i, xk_i| pk_i.* = pk_i.* + xc_i - xk_i; // pk ← Bk⁻¹⋅∇f(xk) + (xc - xk)

            debug.print("  pk = {d: >9.6}\n", .{self.pk});

            for (0..ta, self.iWk[0..ta]) |i, ia| {
                self.tk[i] = self.pk[ia]; // use tk as buffer
            }

            debug.print("  tk = {d: >9.6}\n", .{self.tk[0..ta]});

            try trsv('L', 'N', self.AHA[0..ta], self.tk[0..ta]); // L⋅y = b
            try trsv('L', 'T', self.AHA[0..ta], self.tk[0..ta]); // Lᵀ⋅x = y

            debug.print("  tk = {d: >9.6}\n", .{self.tk[0..ta]}); // Here, get the Lagrange multiplier λₖ

            @memset(self.pk, 0.0);
            for (self.iWk[0..ta], self.tk[0..ta]) |ia, lambda_ia| self.pk[ia] = lambda_ia; // Akᵀ⋅λₖ
            for (self.pk, self.gk) |*pk_i, gk_i| pk_i.* -= gk_i; // Akᵀ⋅λₖ - gk

            // Bk⁻¹⋅(Akᵀ⋅λₖ - gk)
            try trsv('R', 'T', self.Bk, self.pk);
            try trsv('R', 'N', self.Bk, self.pk);

            debug.print("  = = = = =\n", .{});

            debug.print("  pk = {d: >9.6}\n", .{self.pk});

            for (self.tk, self.pk, self.xk, lb, ub) |*tk_i, pk_i, xk_i, lb_i, ub_i| {
                if (0.0 < pk_i and ub_i < math.inf(f64)) {
                    tk_i.* = (ub_i - xk_i) / pk_i;
                } else if (pk_i < 0.0 and -math.inf(f64) < lb_i) {
                    tk_i.* = (lb_i - xk_i) / pk_i;
                } else tk_i.* = math.inf(f64);
                if (tk_i.* == 0.0) tk_i.* = math.inf(f64);
            }

            insertionSort(self.tk);

            debug.print("  line-search step max. = {d: >9.6}\n", .{self.tk[0]});

            // xn ← xk + α⋅pk, gn ← ∇f(xn)
            self.search(obj, null, if (self.tk[0] < math.inf(f64)) self.tk[0] else null, opt) catch |err| {
                switch (err) {
                    Errors.DescentDirectionError => {
                        debug.print("Finished by non-descent direction.\n", .{});
                        break;
                    },
                    else => return err,
                }
            };
        } else {
            // pk ← Bk⁻¹⋅∇f(xk)
            @memcpy(self.pk, self.gk);
            try trsv('R', 'T', self.Bk, self.pk);
            try trsv('R', 'N', self.Bk, self.pk);
            for (self.pk) |*p| p.* = -p.*; // pk ← -Bk⁻¹⋅∇f(xk)

            // xn ← xk + α⋅pk, gn ← ∇f(xn)
            self.search(obj, null, null, opt) catch |err| {
                switch (err) {
                    Errors.DescentDirectionError => {
                        debug.print("Finished by non-descent direction.\n", .{});
                        break;
                    },
                    else => return err,
                }
            };

            ta = 0;
            for (self.xn, lb, ub) |*xn_i, lb_i, ub_i| {
                if (xn_i.* < lb_i) {
                    xn_i.* = lb_i;
                    ta += 1;
                } else if (ub_i < xn_i.*) {
                    xn_i.* = ub_i;
                    ta += 1;
                }
            }

            if (0 < ta) obj.grad(self.xn, self.gn); // gn ← ∇f(xn)
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if (opt.SHOW) debug.print("x{d: <3} = {d: >9.6}, f{d: <3} = {e: <12.10}\n", .{ kx + 1, self.xn, kx + 1, obj.func(self.xn) });

        // sk ← xn - xk
        for (self.sk, self.xn, self.xk) |*sk_i, xn_i, xk_i| sk_i.* = xn_i - xk_i;
        sn = 0.0;
        for (self.sk) |sk_i| sn += pow2(sk_i);
        sn = @sqrt(sn);
        debug.print("  sn = {d}\n", .{sn});
        if (sn <= opt.XTOL) {
            debug.print("Finished by XTOL.\n", .{});
            break;
        }

        // yk ← ∇f(xn) - ∇f(xk) = gn - gk
        for (self.yk, self.gn, self.gk) |*yk_i, gn_i, gk_i| yk_i.* = gn_i - gk_i;
        ym = 0.0;
        for (self.yk) |ym_i| ym = @max(ym, @abs(ym_i));
        debug.print("  ym = {d}\n", .{ym});
        if (ym <= opt.GTOL) {
            debug.print("Finished by GTOL.\n", .{});
            break;
        }

        // secant_norm2 ← skᵀ⋅yk
        const secant_norm2: f64 = try dot(self.sk, self.yk);
        // if (secant_norm2 <= 0.0) return Errors.CurvatureConditionError;

        rk = secant_norm2 / try dot(self.yk, self.yk); // γ = sᵀ⋅y / yᵀ⋅y
        if (2.2e-16 < rk) {
            // sk ← Lkᵀ⋅sk = Rk⋅sk
            try trmv('R', 'N', self.Bk, self.sk);

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
    const SHOW_ITERATIONS: bool = true;

    const page = testing.allocator;

    const n: usize = 2;
    // for (2..6) |n| {
    debug.print("\x1b[32m[[ Line Search Test Case: Rosenbrock's function {d}D ]]\x1b[0m\n", .{n});

    const bfgsb: *Self = try Self.init(page, n);
    defer bfgsb.deinit(page);

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    var lb: [2]f64 = .{ -1.0, 0.0 };
    var ub: [2]f64 = .{ 0.8, 2.0 };
    // for (bfgsb.xk, 1..) |*p, i| p.* = if (i < n) -1.2 else 1.0;
    inline for (bfgsb.xk, .{ -0.3, 1.8 }) |*p, v| p.* = v;

    try bfgsb.solve(rosenbrock, &lb, &ub, .{ .KMAX = 550, .SHOW = SHOW_ITERATIONS });
    // for (bfgsb.xk, 0..) |x, i| {
    //     if (testing.expectApproxEqRel(x, 1.0, 1e-12)) |_| {} else |_| debug.print("x[{d}] = {d}\n", .{ i, x });
    // }
    // }
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

fn insertionSort(arr: []f64) void {
    if (arr.len == 1) return;
    var j: usize = undefined;

    for (arr[1..], 1..) |val, i| {
        j = i;
        while (0 < j and val < arr[j - 1]) : (j -= 1) {
            arr[j] = arr[j - 1];
        }
        arr[j] = val;
    }
    return;
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

const trmv = @import("./trmv.zig").trmv;
const trsv = @import("./trsv.zig").trsv;
const Array = @import("./array.zig").Array;
const cholesky = @import("./cholesky.zig").cholesky;
const Rosenbrock = @import("./Rosenbrock.zig");
