//! References
//! [1] J. Nocedal, S. J. Wright, 2006, "Numerical Optimization 2nd Edition"

xm: []f64, // xₘ
xn: []f64, // xₙ

gm: []f64, // ∇f(xₘ)
gn: []f64, // ∇f(xₙ)

pm: []f64, // pₘ
rs: []f64, // [ ρ(k-1), ρ(k-2), …, ρ(k-m) ]
as: []f64, // [ α(k-1), α(k-2), …, α(k-m) ]

Sm: [][]f64, // [ s(k-1), s(k-2), …, s(k-m) ]
Ym: [][]f64, // [ y(k-1), y(k-2), …, y(k-m) ]

const Self: type = @This();

const Options = struct {
    capacity: ?usize = null,
};

fn init(allocator: mem.Allocator, n: usize, comptime opt: Options) AllocError!*Self {
    const capacity: comptime_int = opt.capacity orelse 30;
    const ArrF64 = Array(f64){ .allocator = allocator };

    const self: *Self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.xm = try ArrF64.vector(n);
    errdefer ArrF64.free(self.xm);

    self.xn = try ArrF64.vector(n);
    errdefer ArrF64.free(self.xn);

    self.gm = try ArrF64.vector(n);
    errdefer ArrF64.free(self.gm);

    self.gn = try ArrF64.vector(n);
    errdefer ArrF64.free(self.gn);

    self.pm = try ArrF64.vector(n);
    errdefer ArrF64.free(self.pm);

    self.rs = try ArrF64.vector(capacity);
    errdefer ArrF64.free(self.rs);

    self.as = try ArrF64.vector(capacity);
    errdefer ArrF64.free(self.as);

    self.Sm = try ArrF64.matrix(capacity, n);
    errdefer ArrF64.free(self.Sm);

    self.Ym = try ArrF64.matrix(capacity, n);

    return self;
}

fn deinit(self: *const Self, allocator: mem.Allocator) void {
    const ArrF64 = Array(f64){ .allocator = allocator };

    ArrF64.free(self.Ym);
    ArrF64.free(self.Sm);
    ArrF64.free(self.as);
    ArrF64.free(self.rs);
    ArrF64.free(self.pm);
    ArrF64.free(self.gn);
    ArrF64.free(self.gm);
    ArrF64.free(self.xn);
    ArrF64.free(self.xm);

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

//
// L-BFGS Routines
//

fn solve(self: *const Self, obj: anytype, comptime opt: Options) !void {
    const capacity: comptime_int = opt.capacity orelse 30;
    var k: usize = 0;

    // 1st Step:
    //     x₀ := self.xm, ∇f(x₀) := self.gm
    //     x₁ := self.xn, ∇f(x₁) := self.gn

    obj.grad(self.xm, self.gm); // gₘ ← ∇f(x₀)
    for (self.pm, self.gm) |*p, g| p.* = -g; // pₘ ← p₀ = -∇f(x₀)

    try self.search(obj, .{}); // x₁ ← x₀ + α⋅p₀, gₙ ← ∇f(x₁)
    for (self.Sm[0], self.xn, self.xm) |*s_i, xn_i, xm_i| s_i.* = xn_i - xm_i; // s₀ ← x₁ - x₀
    for (self.Ym[0], self.gn, self.gm) |*y_i, gn_i, gm_i| y_i.* = gn_i - gm_i; // y₀ ← ∇f(x₁) - ∇f(x₀)
    self.rs[0] = blk: {
        const rho: f64 = 1.0 / dot(self.Sm[0], self.Ym[0]); // ρ₀ ← 1.0 / s₀ᵀ⋅y₀
        if (rho <= 0.0) unreachable;
        break :blk rho;
    };

    @memcpy(self.xm, self.xn); // xₘ ← x₁
    @memcpy(self.gm, self.gn); // gₘ ← ∇f(x₁)

    k += 1;

    while (k < capacity) : (k += 1) {
        // 2nd Step:
        //     x₁ := self.xm, ∇f(x₁) := self.gm
        //     x₂ := self.xn, ∇f(x₂) := self.gn

        @memcpy(self.pm, self.gm); // q ← ∇f(x₁)

        // for i = k-1, k-2, …, k-m
        //     αᵢ ← ρᵢ⋅(sᵢᵀ⋅q)
        //     q  ← q - αᵢ⋅yᵢ
        // end
        for (0..k) |i| {
            self.as[i] = self.rs[i] * dot(self.Sm[i], self.pm);
            for (self.pm, self.Ym[i]) |*q_j, y_j| {
                q_j.* -= self.as[i] * y_j;
            }
        }

        const diag: f64 = (1.0 / self.rs[0]) / dot(self.Ym[0], self.Ym[0]); // γ₁ = s₀ᵀ⋅y₀ / y₀ᵀ⋅y₀

        // r ← γ₁I⋅q, γ₁ = s₀ᵀ⋅y₀ / y₀ᵀ⋅y₀
        for (self.pm) |*p| p.* *= diag;

        // for i = k-m, k-m+q, …, k-1
        //     β ← ρᵢ⋅(yᵢᵀ⋅r)
        //     r ← r + sᵢ⋅(αᵢ - β)
        // end
        var i: usize = k - 1;
        while (true) : (i -= 1) {
            const beta: f64 = self.rs[i] * dot(self.Ym[i], self.pm);
            for (self.pm, self.Sm[i]) |*p, s| {
                p.* += s * (self.as[i] - beta);
            }
            if (i == 0) break;
        }

        for (self.pm) |*p| p.* = -p.*; // pₘ ← -H₁⋅∇f(x₁)
        try self.search(obj, .{}); // x₂ ← x₁ + α⋅p₁, gₙ ← ∇f(x₂)

        for (0..k) |j| {
            @memcpy(self.Sm[j + 1], self.Sm[j]);
            @memcpy(self.Ym[j + 1], self.Ym[j]);
            self.rs[j + 1] = self.rs[j];
        }
        for (self.Sm[0], self.xn, self.xm) |*s_i, xn_i, xm_i| s_i.* = xn_i - xm_i;
        for (self.Ym[0], self.gn, self.gm) |*y_i, gn_i, gm_i| y_i.* = gn_i - gm_i;
        self.rs[0] = blk: {
            const rho: f64 = 1.0 / dot(self.Sm[0], self.Ym[0]); // ρ₀ ← 1.0 / s₀ᵀ⋅y₀
            if (rho <= 0.0) unreachable;
            break :blk rho;
        };

        @memcpy(self.xm, self.xn); // xₘ ← x₁
        @memcpy(self.gm, self.gn); // gₘ ← ∇f(x₁)
    }

    while (k < 150) : (k += 1) {
        // 2nd Step:
        //     x₁ := self.xm, ∇f(x₁) := self.gm
        //     x₂ := self.xn, ∇f(x₂) := self.gn

        @memcpy(self.pm, self.gm); // q ← ∇f(x₁)

        // for i = k-1, k-2, …, k-m
        //     αᵢ ← ρᵢ⋅(sᵢᵀ⋅q)
        //     q  ← q - αᵢ⋅yᵢ
        // end
        for (0..capacity - 1) |i| {
            self.as[i] = self.rs[i] * dot(self.Sm[i], self.pm);
            for (self.pm, self.Ym[i]) |*q_j, y_j| {
                q_j.* -= self.as[i] * y_j;
            }
        }

        const diag: f64 = (1.0 / self.rs[0]) / dot(self.Ym[0], self.Ym[0]); // γ₁ = s₀ᵀ⋅y₀ / y₀ᵀ⋅y₀

        // r ← γ₁I⋅q, γ₁ = s₀ᵀ⋅y₀ / y₀ᵀ⋅y₀
        for (self.pm) |*p| p.* *= diag;

        // for i = k-m, k-m+q, …, k-1
        //     β ← ρᵢ⋅(yᵢᵀ⋅r)
        //     r ← r + sᵢ⋅(αᵢ - β)
        // end
        var i: usize = capacity - 1;
        while (true) : (i -= 1) {
            const beta: f64 = self.rs[i] * dot(self.Ym[i], self.pm);
            for (self.pm, self.Sm[i]) |*p, s| {
                p.* += s * (self.as[i] - beta);
            }
            if (i == 0) break;
        }

        for (self.pm) |*p| p.* = -p.*; // pₘ ← -H₁⋅∇f(x₁)
        try self.search(obj, .{}); // x₂ ← x₁ + α⋅p₁, gₙ ← ∇f(x₂)

        for (0..capacity - 1) |j| {
            @memcpy(self.Sm[j + 1], self.Sm[j]);
            @memcpy(self.Ym[j + 1], self.Ym[j]);
            self.rs[j + 1] = self.rs[j];
        }
        for (self.Sm[0], self.xn, self.xm) |*s_i, xn_i, xm_i| s_i.* = xn_i - xm_i;
        for (self.Ym[0], self.gn, self.gm) |*y_i, gn_i, gm_i| y_i.* = gn_i - gm_i;
        self.rs[0] = blk: {
            const rho: f64 = 1.0 / dot(self.Sm[0], self.Ym[0]); // ρ₀ ← 1.0 / s₀ᵀ⋅y₀
            if (rho <= 0.0) unreachable;
            break :blk rho;
        };

        @memcpy(self.xm, self.xn); // xₘ ← x₁
        @memcpy(self.gm, self.gn); // gₘ ← ∇f(x₁)
    }
}

//
// Unit-testing on Rosenbrock's functions
//

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
    const bfgs: *Self = try Self.init(page, dims, .{});
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.solve(rosenbrock, .{});

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    // try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 3D" {
    const dims: usize = 3;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims, .{});
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.solve(rosenbrock, .{});

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    // try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 4D" {
    const dims: usize = 4;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims, .{});
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.solve(rosenbrock, .{});

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    // try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

test "BFGS Test Case: Rosenbrock's function 5D" {
    const dims: usize = 5;
    debug.print("[[ BFGS Test Case: Rosenbrock's function {d}D ]]\n", .{dims});

    const page = std.testing.allocator;
    const bfgs: *Self = try Self.init(page, dims, .{});
    defer bfgs.deinit(page);

    for (bfgs.xm, 1..) |*p, i| p.* = if (i < dims) -1.2 else 1.0;

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };
    debug.print("  x_now = {d: >7.5}, f_now = {d: >18.15}\n", .{ bfgs.xm, rosenbrock.func(bfgs.xm) });

    // xₙ ← xₘ + α⋅pₘ
    try bfgs.solve(rosenbrock, .{});

    debug.print("  x_new = {d: >7.5}, f_new = {d: >18.15}\n", .{ bfgs.xn, rosenbrock.func(bfgs.xn) });

    // try testing.expect(rosenbrock.func(bfgs.xn) < rosenbrock.func(bfgs.xm));
}

//
// Subroutines
//

const AllocError = mem.Allocator.Error;

fn Array(comptime T: type) type {
    // already comptime scope
    const slice_al: comptime_int = @alignOf([]T);
    const child_al: comptime_int = @alignOf(T);
    const slice_sz: comptime_int = @sizeOf(usize) * 2;
    const child_sz: comptime_int = @sizeOf(T);

    return struct {
        allocator: std.mem.Allocator,

        fn matrix(self: @This(), nrow: usize, ncol: usize) AllocError![][]T {
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

        fn vector(self: @This(), n: usize) AllocError![]T {
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
const math = std.math;
const debug = std.debug;
const testing = std.testing;
