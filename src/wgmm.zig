//! 1D Weighted Gaussian Mixture Model

const slice_al: comptime_int = @alignOf([]f64);
const child_al: comptime_int = @alignOf(f64);
const slice_sz: comptime_int = @sizeOf(usize) * 2;
const child_sz: comptime_int = @sizeOf(f64);

const Error = error{DimensionMismatch};

pub inline fn respAllocSize(n: usize, k: usize) usize {
    return n * k * child_sz + n * slice_sz;
}

pub fn respSlice(buff: []u8, n: usize, k: usize) ![][]f64 {
    if (buff.len != respAllocSize(n, k)) return error.DimensionMismatch;

    const mat: [][]f64 = blk: {
        const ptr: [*]align(slice_al) []f64 = @ptrCast(@alignCast(buff.ptr));
        break :blk ptr[0..n];
    };

    const chunk_sz: usize = k * child_sz;
    var padding: usize = n * slice_sz;

    for (mat) |*row| {
        row.* = blk: {
            const ptr: [*]align(child_al) f64 = @ptrCast(@alignCast(buff.ptr + padding));
            break :blk ptr[0..k];
        };
        padding += chunk_sz;
    }

    return mat;
}

// x: []f64, x.len == N <- data points
// m: []f64, m.len == K <- K-means
// s: []f64, s.len == K <- K-std. deviations
// p: []f64, p.len == K <- K-priors
// r: [][]f64, r.len == N, r[n].len == K <- responsibility matrix
pub fn stepE(x: []f64, w: []f64, N: usize, m: []f64, s: []f64, p: []f64, K: usize, r: [][]f64, L: *f64) void {
    for (0..N) |n| {
        for (0..K) |k| {
            r[n][k] = @log(p[k]) - @log(s[k]) - 0.5 * pow2((x[n] - m[k]) / s[k]) - LOG_SQRT_TWOPI;
        }
    }

    var znmax: f64 = undefined;
    var temp: f64 = undefined;

    L.* = 0.0;

    for (0..N) |n| {
        znmax = -INF;
        for (0..K) |k| {
            if (r[n][k] > znmax) znmax = r[n][k];
        }

        temp = 0.0;
        for (0..K) |k| temp += @exp(r[n][k] - znmax);

        temp = znmax + @log(temp);
        for (0..K) |k| r[n][k] = @exp(r[n][k] - temp);

        L.* += w[n] * temp;
    }

    return;
}

pub fn stepM(x: []f64, w: []f64, N: usize, m: []f64, s: []f64, p: []f64, K: usize, r: [][]f64) void {
    var wtot: f64 = 0.0; // sum of weights
    for (0..N) |n| wtot += w[n];

    var N_k: f64 = undefined;
    var tmp: f64 = undefined;

    for (0..K) |k| {
        N_k = 0.0;
        for (0..N) |n| N_k += w[n] * r[n][k];

        p[k] = N_k / wtot;

        tmp = 0.0;
        for (0..N) |n| tmp += r[n][k] * w[n] * x[n];

        m[k] = tmp / N_k;

        tmp = 0.0;
        for (0..N) |n| tmp += r[n][k] * w[n] * pow2(x[n] - m[k]);

        s[k] = @sqrt(tmp / N_k);
    }

    return;
}

inline fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");

const INF: comptime_float = std.math.inf(f64);
const LOG_SQRT_TWOPI: comptime_float = @log(@sqrt(2.0 * std.math.pi));
