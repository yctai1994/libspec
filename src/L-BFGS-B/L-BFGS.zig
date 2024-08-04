//! References
//! [1] J. Nocedal, S. J. Wright, 2006, "Numerical Optimization 2nd Edition"

xm: []f64 = undefined, // xₘ
xn: []f64 = undefined, // xₙ

gm: []f64 = undefined, // ∇f(xₘ)
gn: []f64 = undefined, // ∇f(xₙ)

pm: []f64 = undefined, // pₘ

const AllocError = mem.Allocator.Error;

const Self: type = @This();

fn init(allocator: mem.Allocator, n: usize) AllocError!*Self {
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

    self.pm = try allocator.alloc(f64, n);

    return self;
}

fn deinit(self: *const Self, allocator: mem.Allocator) void {
    allocator.free(self.pm);
    allocator.free(self.gn);
    allocator.free(self.gm);
    allocator.free(self.xn);
    allocator.free(self.xm);

    allocator.destroy(self);
}

//
// Line Search Algorithm for the Wolfe Conditions
//

const StrongWolfe = struct {
    c1: comptime_float = 1e-4,
    c2: comptime_float = 0.9,
    a_max: comptime_float = 65536.0,
    a_min: comptime_float = 0.0,
};

const LineSearchError = error{
    SearchError,
    ZoomError,
};

// Algorithm 3.5 in [1]
fn search(self: *const Self, obj: anytype, comptime param: StrongWolfe) LineSearchError!void {
    obj.grad(self.xm, self.gm); // ∇f(xₘ)
    for (self.pm, self.gm) |*p, g| p.* = -g; // pₘ ← -∇f(xₘ)

    const f0: f64 = obj.func(self.xm); // ϕ(0) = f(xₘ)
    const g0: f64 = dot(self.pm, self.gm); // ϕ'(0) = pₘᵀ⋅∇f(xₘ)

    if (0.0 < g0) return LineSearchError.SearchError;

    var f_old: f64 = undefined;
    var f_now: f64 = undefined;
    var g_now: f64 = undefined;

    var a_old: f64 = param.a_min;
    var a_now: f64 = 1e-4;

    var iter: usize = 0;

    while (a_now < param.a_max) : (iter += 1) {
        for (self.xn, self.xm, self.pm) |*xn_i, xm_i, pm_i| xn_i.* = xm_i + a_now * pm_i; // xₜ ← xₘ + α⋅pₘ
        f_now = obj.func(self.xn); // ϕ(α) = f(xₘ + α⋅pₘ)

        // Test Wolfe conditions
        if ((f_now > f0 + param.c1 * a_now * g0) or (iter > 0 and f_now > f_old)) {
            return try self.zoom(a_old, a_now, f0, g0, obj, param);
        }

        obj.grad(self.xn, self.gn); // ∇f(xₘ + α⋅pₘ)
        g_now = dot(self.pm, self.gn); // ϕ'(α) = pₘᵀ⋅∇f(xₘ + α⋅pₘ)

        if (@abs(g_now) <= -param.c2 * g0) break; // return a_now
        if (0.0 <= g_now) return try self.zoom(a_now, a_old, f0, g0, obj, param);

        a_old = a_now;
        f_old = f_now;
        a_now = 2.0 * a_now;
    } else return LineSearchError.SearchError;
}

// Algorithm 3.6 in [1]
fn zoom(self: *const Self, a_lb: f64, a_rb: f64, f0: f64, g0: f64, obj: anytype, comptime param: StrongWolfe) LineSearchError!void {
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

    for (self.xn, self.xm, self.pm) |*x_lo, xm_i, sm_i| x_lo.* = xm_i + a_lo * sm_i; // xₜ ← xₘ + α_lo⋅pₘ
    obj.grad(self.xn, self.gn); // ∇f(xₘ + α_lo⋅pₘ)

    f_lo = obj.func(self.xn); // ϕ(α_lo) = f(xₘ + α_lo⋅pₘ)
    g_lo = dot(self.pm, self.gn); // ϕ'(α_lo) = pₘᵀ⋅∇f(xₘ + α_lo⋅pₘ)

    for (self.xn, self.xm, self.pm) |*x_hi, xm_i, sm_i| x_hi.* = xm_i + a_hi * sm_i; // xₜ ← xₘ + α_hi⋅pₘ
    obj.grad(self.xn, self.gn); // ∇f(xₘ + α_hi⋅pₘ)

    f_hi = obj.func(self.xn); // ϕ(α_hi) = f(xₘ + α_hi⋅pₘ)
    g_hi = dot(self.pm, self.gn); // ϕ'(α_hi) = pₘᵀ⋅∇f(xₘ + α_hi⋅pₘ), ϕ'(α_hi) can be positive

    while (iter < 10) : (iter += 1) {
        // Interpolate α
        a_now = if (a_lo < a_hi)
            try interpolate(a_lo, a_hi, f_lo, f_hi, g_lo, g_hi)
        else
            try interpolate(a_hi, a_lo, f_hi, f_lo, g_hi, g_lo);

        for (self.xn, self.xm, self.pm) |*xn_i, xm_i, sm_i| xn_i.* = xm_i + a_now * sm_i; // xₜ ← xₘ + α⋅pₘ
        obj.grad(self.xn, self.gn); // ∇f(xₘ + α⋅pₘ)
        f_now = obj.func(self.xn); // ϕ(α) = f(xₘ + α⋅pₘ)
        g_now = dot(self.pm, self.gn); // ϕ'(α) = pₘᵀ⋅∇f(xₘ + α⋅pₘ)

        if ((f_now > f0 + param.c1 * a_now * g0) or (f_now > f_lo)) {
            a_hi = a_now;
            f_hi = f_now;
            g_hi = g_now;
        } else {
            if (@abs(g_now) <= -param.c2 * g0) break; // return a_now
            if (0.0 <= g_now * (a_hi - a_lo)) {
                a_hi = a_lo;
                f_hi = f_lo;
                g_hi = g_lo;
            }

            a_lo = a_now;
            f_lo = f_now;
            g_lo = g_now;
        }
    } else return LineSearchError.ZoomError;
}

// Equation 3.59 in [1]
fn interpolate(a_old: f64, a_new: f64, f_old: f64, f_new: f64, g_old: f64, g_new: f64) LineSearchError!f64 {
    if (a_new <= a_old) return LineSearchError.ZoomError;
    const d1: f64 = g_old + g_new - 3.0 * (f_old - f_new) / (a_old - a_new);
    const d2: f64 = @sqrt(d1 * d1 - g_old * g_new);
    const nu: f64 = g_new + d2 - d1;
    const de: f64 = g_new - g_old + 2.0 * d2;
    return a_new - (a_new - a_old) * (nu / de);
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

test "BFGS Test Case: Rosenbrock's function 2D" {
    const dims: usize = 2;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims);
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.search(rosenbrock, .{});

    // gₙ ← ∇f(xₙ)
    rosenbrock.grad(bfgs.xn, bfgs.gn);

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 3D" {
    const dims: usize = 3;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims);
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.search(rosenbrock, .{});

    // gₙ ← ∇f(xₙ)
    rosenbrock.grad(bfgs.xn, bfgs.gn);

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 4D" {
    const dims: usize = 4;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims);
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.search(rosenbrock, .{});

    // gₙ ← ∇f(xₙ)
    rosenbrock.grad(bfgs.xn, bfgs.gn);

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 5D" {
    const dims: usize = 5;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims);
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.search(rosenbrock, .{});

    // gₙ ← ∇f(xₙ)
    rosenbrock.grad(bfgs.xn, bfgs.gn);

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
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
