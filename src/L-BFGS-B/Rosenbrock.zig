a: f64,
b: f64,

const Self: type = @This();

pub fn func(self: *const Self, x: []f64) f64 {
    const n: usize = x.len;
    var val: f64 = 0.0;
    for (0..n - 1) |i| val += self.subfunc(x[i], x[i + 1]);
    return val;
}

fn subfunc(self: *const Self, x: f64, y: f64) f64 {
    return pow2(self.a - x) + self.b * pow2(y - pow2(x));
}

pub fn grad(self: *const Self, x: []f64, g: []f64) void {
    const n: usize = x.len;
    if (n != g.len) unreachable;
    const m: usize = n - 1;

    g[0] = self.subgrad1(x[0], x[1]);
    for (1..m) |i| g[i] = self.subgrad1(x[i], x[i + 1]) + self.subgrad2(x[i - 1], x[i]);
    g[m] = self.subgrad2(x[m - 1], x[m]);
}

fn subgrad1(self: *const Self, x: f64, y: f64) f64 {
    return 2.0 * (x - self.a) + 4.0 * self.b * x * (pow2(x) - y);
}

fn subgrad2(self: *const Self, x: f64, y: f64) f64 {
    return 2.0 * self.b * (y - pow2(x));
}

inline fn pow2(x: f64) f64 {
    return x * x;
}
