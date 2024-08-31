const Error = error{DimensionMismatch};

pub fn linearFit(xdat: []f64, ydat: []f64, wvec: []f64) !void {
    const n: usize = xdat.len;
    if (n != ydat.len) return error.DimensionMismatch;

    var Xmat: [2][2]f64 = undefined;
    var Amat: [2][2]f64 = undefined;

    //
    //     ┌              ┐      ┌       ┐
    //     │ 1⃗ₙ⋅1⃗ₙ  1⃗ₙ⋅x⃗ₙ │      │ 1⃗ₙ⋅y⃗ₙ │
    // 𝐀 = │              │, 𝐛 = │       │
    //     │ x⃗ₙ⋅1⃗ₙ  x⃗ₙ⋅x⃗ₙ │      │ x⃗ₙ⋅y⃗ₙ │
    //     └              ┘      └       ┘
    //

    Xmat[0][0] = @floatFromInt(n);
    Xmat[0][1] = sum(xdat);
    Xmat[1][0] = Xmat[0][1];
    Xmat[1][1] = dot(xdat, xdat);

    const detA: f64 = Xmat[0][0] * Xmat[1][1] - Xmat[0][1] * Xmat[1][0];

    Amat[0][0] = Xmat[1][1] / detA;
    Amat[1][1] = Xmat[0][0] / detA;
    Amat[0][1] = -(Xmat[0][1] / detA);
    Amat[1][0] = -(Xmat[1][0] / detA);

    const bvec0: f64 = sum(ydat);
    const bvec1: f64 = dot(xdat, ydat);

    wvec[0] = Amat[0][0] * bvec0 + Amat[0][1] * bvec1;
    wvec[1] = Amat[1][0] * bvec0 + Amat[1][1] * bvec1;
}

fn dot(x: []f64, y: []f64) f64 {
    var ret: f64 = 0.0;
    for (x, y) |x_i, y_i| ret += x_i * y_i;
    return ret;
}

fn sum(x: []f64) f64 {
    var ret: f64 = 0.0;
    for (x) |x_i| ret += x_i;
    return ret;
}
