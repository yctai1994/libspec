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

const AllocError = mem.Allocator.Error;

const Self: type = @This();

pub fn init(allocator: mem.Allocator, n: usize) AllocError!*Self {
    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

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

    self.Bm = try Hess.init(allocator, n);

    return self;
}

pub fn deinit(self: *const Self, allocator: mem.Allocator) void {
    self.Bm.deinit(allocator);

    allocator.free(self.vm);
    allocator.free(self.um);
    allocator.free(self.am);
    allocator.free(self.ym);
    allocator.free(self.sm);
    allocator.free(self.gn);
    allocator.free(self.gm);
    allocator.free(self.xn);
    allocator.free(self.xm);

    allocator.destroy(self);
}

fn search(self: *const Self, obj: anytype) void {
    const c1: comptime_float = 1e-4;
    const c2: comptime_float = 0.9;

    const amax: comptime_float = 1.0;
    const amin: comptime_float = 0.0;

    const phi_0: f64 = obj.func(self.xm); // ϕ(0) = f(xₘ)
    obj.grad(self.xm, self.gm); // ∇f(xₘ)
    for (self.sm, self.gm) |*p, d| p.* = -d; // pₘ ← -∇f(xₘ)
    const dphi_0: f64 = dot(self.sm, self.gm); // ϕ'(0) = pₘᵀ⋅∇f(xₘ)

    // debug.print("phi_0 = {d}, dphi_0 = {d}\n", .{ phi_0, dphi_0 });

    if (0.0 < dphi_0) unreachable;

    var phi_old: f64 = undefined;
    var phi: f64 = undefined;
    var dphi: f64 = undefined;

    var a_old: f64 = amin;
    var a: f64 = 0.5 * (amin + amax);

    var iter: usize = 0;

    while (iter < 20) : (iter += 1) {
        for (self.xn, self.xm, self.sm) |*xn_i, xm_i, sm_i| xn_i.* = xm_i + a * sm_i; // xₜ ← xₘ + α⋅pₘ
        phi = obj.func(self.xn); // ϕ(α) = f(xₘ + α⋅pₘ)

        // Test Wolfe conditions
        if ((phi > phi_0 + c1 * a * dphi_0) or (iter > 0 and phi > phi_old)) {
            return self.zoom(a_old, a, phi_0, dphi_0, obj);
        }

        obj.grad(self.xn, self.gn); // ∇f(xₘ + α⋅pₘ)
        dphi = dot(self.sm, self.gn); // ϕ'(α) = pₘᵀ⋅∇f(xₘ + α⋅pₘ)

        // debug.print("phi = {d}, dphi = {d}, a = {d}\n", .{ phi, dphi, a });

        if (@abs(dphi) <= -c2 * dphi_0) break; // return a;
        if (0.0 <= dphi) return self.zoom(a, a_old, phi_0, dphi_0, obj);

        a_old = a;
        phi_old = phi;
        a = 0.5 * (a + amax);
    } else {
        iter = 0;
        while (iter < 30) : (iter += 1) {
            for (self.xn, self.xm, self.sm) |*xn_i, xm_i, sm_i| xn_i.* = xm_i + a * sm_i; // xₜ ← xₘ + α⋅pₘ
            phi = obj.func(self.xn); // ϕ(α) = f(xₘ + α⋅pₘ)

            // Test Wolfe conditions
            if ((phi > phi_0 + c1 * a * dphi_0) or (iter > 0 and phi > phi_old)) {
                return self.zoom(a_old, a, phi_0, dphi_0, obj);
            }

            obj.grad(self.xn, self.gn); // ∇f(xₘ + α⋅pₘ)
            dphi = dot(self.sm, self.gn); // ϕ'(α) = pₘᵀ⋅∇f(xₘ + α⋅pₘ)

            // debug.print("phi = {d}, dphi = {d}, a = {d}\n", .{ phi, dphi, a });

            if (@abs(dphi) <= -c2 * dphi_0) break; // return a;
            if (0.0 <= dphi) return self.zoom(a, a_old, phi_0, dphi_0, obj);

            a_old = a;
            phi_old = phi;
            a *= 2.0;
        } else unreachable;
    }
}

// it's possible that a_lb > a_rb
fn zoom(self: *const Self, a_lb: f64, a_rb: f64, phi_0: f64, dphi_0: f64, obj: anytype) void {
    const c1: comptime_float = 1e-4;
    const c2: comptime_float = 0.9;

    var phi_lo: f64 = undefined;
    var phi_hi: f64 = undefined;

    var dphi_lo: f64 = undefined;
    var dphi_hi: f64 = undefined;

    var a_lo: f64 = a_lb;
    var a_hi: f64 = a_rb;

    var a: f64 = undefined;
    var phi: f64 = undefined;
    var dphi: f64 = undefined;

    var iter: usize = 0;

    for (self.xn, self.xm, self.sm) |*x_lo, xm_i, sm_i| x_lo.* = xm_i + a_lo * sm_i; // xₜ ← xₘ + α_lo⋅pₘ
    obj.grad(self.xn, self.gn); // ∇f(xₘ + α_lo⋅pₘ)
    phi_lo = obj.func(self.xn); // ϕ(α_lo) = f(xₘ + α_lo⋅pₘ)
    dphi_lo = dot(self.sm, self.gn); // ϕ'(α_lo) = pₘᵀ⋅∇f(xₘ + α_lo⋅pₘ)

    for (self.xn, self.xm, self.sm) |*x_hi, xm_i, sm_i| x_hi.* = xm_i + a_hi * sm_i; // xₜ ← xₘ + α_hi⋅pₘ
    obj.grad(self.xn, self.gn); // ∇f(xₘ + α_hi⋅pₘ)
    phi_hi = obj.func(self.xn); // ϕ(α_hi) = f(xₘ + α_hi⋅pₘ)
    dphi_hi = dot(self.sm, self.gn); // ϕ'(α_hi) = pₘᵀ⋅∇f(xₘ + α_hi⋅pₘ), ϕ'(α_hi) can be positive

    while (iter < 100) : (iter += 1) {
        // Interpolate α
        a = if (a_lo < a_hi)
            interpolate(a_lo, a_hi, phi_lo, phi_hi, dphi_lo, dphi_hi)
        else
            interpolate(a_hi, a_lo, phi_hi, phi_lo, dphi_hi, dphi_lo);

        for (self.xn, self.xm, self.sm) |*xn_i, xm_i, sm_i| xn_i.* = xm_i + a * sm_i; // xₜ ← xₘ + α⋅pₘ
        phi = obj.func(self.xn); // ϕ(α) = f(xₘ + α⋅pₘ)
        obj.grad(self.xn, self.gn); // ∇f(xₘ + α⋅pₘ)
        dphi = dot(self.sm, self.gn); // ϕ'(α) = pₘᵀ⋅∇f(xₘ + α⋅pₘ)

        if ((phi > phi_0 + c1 * a * dphi_0) or (phi > phi_lo)) {
            a_hi = a;
            phi_hi = phi;
            dphi_hi = dphi;
        } else {
            if (@abs(dphi) <= -c2 * dphi_0) break; // return a;
            if (0.0 <= dphi * (a_hi - a_lo)) {
                a_hi = a_lo;
                phi_hi = phi_lo;
                dphi_hi = dphi_lo;
            }

            a_lo = a;
            phi_lo = phi;
            dphi_lo = dphi;
        }
    } // else unreachable;
}

fn interpolate(a_old: f64, a_now: f64, phi_old: f64, phi_now: f64, dphi_old: f64, dphi_now: f64) f64 {
    if (!(a_old < a_now)) unreachable;
    const d1: f64 = dphi_old + dphi_now - 3.0 * (phi_old - phi_now) / (a_old - a_now);
    const d2: f64 = @sqrt(d1 * d1 - dphi_old * dphi_now);
    const nu: f64 = dphi_now + d2 - d1;
    const de: f64 = dphi_now - dphi_old + 2.0 * d2;
    return a_now - (a_now - a_old) * (nu / de);
}

const Termination = struct {
    xtol: ?f64 = null,
    gtol: ?f64 = null,
};

pub fn firstStep(self: *const Self, obj: anytype, comptime term: Termination) !bool {
    const xtol: comptime_float = comptime term.xtol orelse math.floatEps(f64);
    const gtol: comptime_float = comptime term.gtol orelse math.floatEps(f64);

    // xₙ ← xₘ + α⋅pₘ
    self.search(obj);

    // gₙ ← ∇f(xₙ)
    obj.grad(self.xn, self.gn);

    // sₘ ← xₙ - xₘ
    for (self.sm, self.xn, self.xm) |*sm_i, xn_i, xm_i| sm_i.* = xn_i - xm_i;
    var snrm: f64 = 0.0;
    for (self.sm) |sm_i| snrm += pow2(sm_i);
    snrm = @sqrt(snrm);
    if (snrm <= xtol) return true;

    // yₘ ← ∇f(xₙ) - ∇f(xₘ) = gₙ - gₘ
    for (self.ym, self.gn, self.gm) |*ym_i, gn_i, gm_i| ym_i.* = gn_i - gm_i;
    var ymax: f64 = 0.0;
    for (self.ym) |ym_i| ymax = @max(ymax, @abs(ym_i));
    if (ymax <= gtol) return true;

    // secant_norm2 ← yₘᵀ⋅sₘ
    var secant_norm2: f64 = dot(self.ym, self.sm);
    if (secant_norm2 <= 0.0) unreachable;

    // √(yₘᵀ⋅yₘ / yₘᵀ⋅sₘ)
    const diag: f64 = @sqrt(dot(self.ym, self.ym) / secant_norm2);
    for (self.Bm.matrix, 0..) |Bm_i, i| {
        // @memset(Bm_i[i..], 0.0);
        @memset(Bm_i, 0.0);
        Bm_i[i] = diag;
    }

    // sₘ ← Lₘᵀ⋅sₘ = Rₘ⋅sₘ
    self.Bm.dtrmv(self.sm);

    // sₘᵀ⋅(Lₘ⋅Lₘᵀ)⋅sₘ ← sₘᵀ⋅sₘ
    const quadratic_form: f64 = dot(self.sm, self.sm);

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

    // αₘ ← √(secant_norm2 / quadratic_form)
    const alpha_m: f64 = @sqrt(secant_norm2 / quadratic_form);

    // aₘ ← αₘ⋅Lₘᵀ⋅sₘ = αₘ⋅Rₘ⋅sₘ
    for (self.am, self.sm) |*am_i, sm_i| am_i.* = alpha_m * sm_i;

    // ‖aₘ‖ ← √(rₘᵀ⋅sₘ)
    const secant_norm: f64 = @sqrt(secant_norm2);

    // uₘ ← aₘ / ‖aₘ‖
    for (self.um, self.am) |*um_i, am_i| um_i.* = am_i / secant_norm;

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

    try self.Bm.update(self.um, self.vm);

    debug.print("Hess:\n", .{});
    for (self.Bm.matrix) |hess_row| {
        debug.print("    {d}\n", .{hess_row});
    }
    // debug.print("\n", .{});

    @memcpy(self.xm, self.xn);
    @memcpy(self.gm, self.gn);

    return false;
}

pub fn iterate(self: *const Self, obj: anytype, comptime term: Termination) !bool {
    const xtol: comptime_float = comptime term.xtol orelse math.floatEps(f64);
    const gtol: comptime_float = comptime term.gtol orelse math.floatEps(f64);
    _ = .{ xtol, gtol };

    // gₘ ← ∇f(xₘ)
    obj.grad(self.xm, self.gm);
    // debug.print("  gm: {d}\n", .{self.gm});

    // pₘ ← Bₘ⁻¹⋅∇f(xₘ), store `pₘ` in `sₘ`
    // pₘ := sₘ, ∇f(xₘ) := gₘ
    self.Bm.solve(self.sm, self.gm);
    // debug.print("  sm: {d}\n", .{self.sm});

    // xₙ ← xₘ - pₘ
    for (self.xn, self.xm, self.sm) |*xn_i, xm_i, pc_i| xn_i.* = xm_i - pc_i;
    // debug.print("  xn: {d}\n", .{self.xn});

    // gₙ ← ∇f(xₙ)
    obj.grad(self.xn, self.gn);

    for (self.gn) |gn_i| {
        if (math.isNan(gn_i)) {
            debug.print(" \x1b[31mnan: use line search\x1b[0m\n", .{});
            // xₙ ← xₘ + α⋅pₘ
            self.search(obj);

            // gₙ ← ∇f(xₙ)
            obj.grad(self.xn, self.gn);
            break;
        }
    }
    // debug.print("  xn: {d}\n", .{self.xn});

    // sₘ ← xₙ - xₘ
    for (self.sm, self.xn, self.xm) |*sm_i, xn_i, xm_i| sm_i.* = xn_i - xm_i;
    // debug.print("  sm: {d}\n", .{self.sm});
    var snrm: f64 = 0.0;
    for (self.sm) |sm_i| snrm += pow2(sm_i);
    snrm = @sqrt(snrm);
    // if (snrm <= xtol) return true;
    // if (snrm <= xtol) return false;

    // yₘ ← ∇f(xₙ) - ∇f(xₘ) = gₙ - gₘ
    for (self.ym, self.gn, self.gm) |*ym_i, gn_i, gm_i| ym_i.* = gn_i - gm_i;
    // debug.print("  ym: {d}, gn: {d}, gm: {d}\n", .{ self.ym, self.gn, self.gm });
    var ymax: f64 = 0.0;
    for (self.ym) |ym_i| ymax = @max(ymax, @abs(ym_i));
    // if (ymax <= gtol) return true;
    // if (ymax <= gtol) return false;

    // secant_norm2 ← yₘᵀ⋅sₘ
    var secant_norm2: f64 = dot(self.ym, self.sm);
    if (secant_norm2 <= 0.0) { // unreachable;
        debug.print(" \x1b[31mnan: yₘᵀ⋅sₘ <= 0.0, use line search\x1b[0m\n", .{});
        // return self.firstStep(obj, term);
        // xₙ ← xₘ + α⋅pₘ
        self.search(obj);

        // gₙ ← ∇f(xₙ)
        obj.grad(self.xn, self.gn);

        // sₘ ← xₙ - xₘ
        for (self.sm, self.xn, self.xm) |*sm_i, xn_i, xm_i| sm_i.* = xn_i - xm_i;
        snrm = 0.0;
        for (self.sm) |sm_i| snrm += pow2(sm_i);
        snrm = @sqrt(snrm);
        // if (snrm <= xtol) return true;

        // yₘ ← ∇f(xₙ) - ∇f(xₘ) = gₙ - gₘ
        for (self.ym, self.gn, self.gm) |*ym_i, gn_i, gm_i| ym_i.* = gn_i - gm_i;
        ymax = 0.0;
        for (self.ym) |ym_i| ymax = @max(ymax, @abs(ym_i));
        // if (ymax <= gtol) return true;

        // secant_norm2 ← yₘᵀ⋅sₘ
        secant_norm2 = dot(self.ym, self.sm);
    }

    // sₘ ← Lₘᵀ⋅sₘ = Rₘ⋅sₘ
    self.Bm.dtrmv(self.sm);

    // sₘᵀ⋅(Lₘ⋅Lₘᵀ)⋅sₘ ← sₘᵀ⋅sₘ
    const quadratic_form: f64 = dot(self.sm, self.sm);

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
    debug.print("theta_m = {d}\n", .{theta_m});

    // αₘ ← √(secant_norm2 / quadratic_form)
    const alpha_m: f64 = @sqrt(secant_norm2 / quadratic_form);

    // aₘ ← αₘ⋅Lₘᵀ⋅sₘ = αₘ⋅Rₘ⋅sₘ
    for (self.am, self.sm) |*am_i, sm_i| am_i.* = alpha_m * sm_i;

    // ‖aₘ‖ ← √(rₘᵀ⋅sₘ)
    const secant_norm: f64 = @sqrt(secant_norm2);

    // uₘ ← aₘ / ‖aₘ‖
    for (self.um, self.am) |*um_i, am_i| um_i.* = am_i / secant_norm;

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

    try self.Bm.update(self.um, self.vm);

    debug.print("Hess:\n", .{});
    for (self.Bm.matrix) |hess_row| {
        debug.print("    {d}\n", .{hess_row});
    }

    @memcpy(self.xm, self.xn);
    @memcpy(self.gm, self.gn);

    return false;
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

// test "BFGS Test Case: Rosenbrock's function 2D" {
//     const page = std.testing.allocator;
//
//     const bfgs: *Self = try Self.init(page, 2);
//     defer bfgs.deinit(page);
//
//     inline for (.{ -1.2, 1.0 }, bfgs.xm) |v, *p| p.* = v;
//
//     const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
//
//     var terminated: bool = try bfgs.firstStep(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//     while (!terminated) terminated = try bfgs.iterate(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//
//     const y: [2]f64 = .{ 1.0, 1.0 };
//     try testing.expect(mem.eql(f64, &y, bfgs.xm));
// }
//
// test "BFGS Test Case: Rosenbrock's function 3D" {
//     const page = std.testing.allocator;
//
//     const bfgs: *Self = try Self.init(page, 3);
//     defer bfgs.deinit(page);
//
//     inline for (.{ -1.2, -1.2, 1.0 }, bfgs.xm) |v, *p| p.* = v;
//
//     const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
//
//     var terminated: bool = try bfgs.firstStep(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//     while (!terminated) terminated = try bfgs.iterate(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//
//     const y: [3]f64 = .{ 1.0, 1.0, 1.0 };
//     try testing.expect(mem.eql(f64, &y, bfgs.xm));
// }
//
// test "BFGS Test Case: Rosenbrock's function 4D" {
//     const page = std.testing.allocator;
//
//     const bfgs: *Self = try Self.init(page, 4);
//     defer bfgs.deinit(page);
//
//     inline for (.{ -1.2, -1.2, -1.2, 1.0 }, bfgs.xm) |v, *p| p.* = v;
//
//     const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
//
//     var terminated: bool = try bfgs.firstStep(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//     while (!terminated) terminated = try bfgs.iterate(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//
//     const y: [4]f64 = .{ 1.0, 1.0, 1.0, 1.0 };
//     try testing.expect(mem.eql(f64, &y, bfgs.xm));
// }
//
// test "BFGS Test Case: Rosenbrock's function 5D" {
//     const page = std.testing.allocator;
//
//     const bfgs: *Self = try Self.init(page, 5);
//     defer bfgs.deinit(page);
//
//     inline for (.{ -1.2, -1.2, -1.2, -1.2, 1.0 }, bfgs.xm) |v, *p| p.* = v;
//
//     const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
//
//     var terminated: bool = try bfgs.firstStep(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//     while (!terminated) terminated = try bfgs.iterate(rosenbrock, .{ .xtol = 1e-16, .gtol = 1e-16 });
//     // debug.print("{d}, {d}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });
//
//     const y: [5]f64 = .{ 1.0, 1.0, 1.0, 1.0, 1.0 };
//     try testing.expect(mem.eql(f64, &y, bfgs.xm));
// }

const Hess = struct {
    buffer: []f64 = undefined,
    matrix: [][]f64 = undefined,

    const slice_al: comptime_int = @alignOf([]f64);
    const child_al: comptime_int = @alignOf(f64);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(f64);

    const HessError = error{SingularError};

    fn init(allocator: mem.Allocator, n: usize) AllocError!*Hess {
        const self: *Hess = try allocator.create(Hess);
        errdefer allocator.destroy(self);

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

        return self;
    }

    fn deinit(self: *const Hess, allocator: mem.Allocator) void {
        const n: usize = self.buffer.len;
        const ptr: [*]u8 = @ptrCast(@alignCast(self.matrix.ptr));
        const len: usize = n * n * child_sz + n * slice_sz;

        allocator.free(self.buffer);
        allocator.free(ptr[0..len]);

        allocator.destroy(self);
    }

    fn update(self: *const Hess, u: []f64, v: []f64) HessError!void {
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
                self.buffer[i] = @abs(self.buffer[i]) * @sqrt(1.0 + pow2(self.buffer[i + 1] / self.buffer[i]));
            } else {
                self.buffer[i] = @abs(self.buffer[i + 1]) * @sqrt(1.0 + pow2(self.buffer[i] / self.buffer[i + 1]));
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

        for (self.matrix, 0..) |R_j, j| {
            if (R_j[j] == 0.0) return HessError.SingularError;
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
    }
};

test "Hess.update" {
    const page = std.testing.allocator;

    const hess: *Hess = try Hess.init(page, 3);
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

    const hess: *Hess = try Hess.init(page, 3);
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
const math = std.math;
const debug = std.debug;
const testing = std.testing;
