const std = @import("std");

const findRoot = @import("./newton.zig").findRoot;

export fn c_find_root(
    fcall: *const fn (x: f64) callconv(.C) f64,
    deriv: *const fn (x: f64) callconv(.C) f64,
    target: f64,
    xleft: f64,
    xright: f64,
    ITMAX: usize, // Maximum allowed number of iterations.
) f64 {
    return findRoot(
        .C,
        fcall,
        deriv,
        target,
        xleft,
        xright,
        ITMAX,
    ) catch std.math.nan(f64);
}

const wgmm = @import("./wgmm.zig");

export fn resp_alloc_size(n: usize, k: usize) usize {
    return wgmm.respAllocSize(n, k);
}

export fn solve_wgmm(
    xptr: [*]f64,
    wptr: [*]f64,
    numN: usize,
    mptr: [*]f64,
    sptr: [*]f64,
    pptr: [*]f64,
    numK: usize,
    buff: [*]u8,
    bufN: usize,
    IMAX: usize,
    INFO: *usize,
) void {
    const xvec: []f64 = xptr[0..numN];
    const wvec: []f64 = wptr[0..numN];
    const mvec: []f64 = mptr[0..numK];
    const svec: []f64 = sptr[0..numK];
    const pvec: []f64 = pptr[0..numK];

    const resp: [][]f64 = wgmm.respSlice(buff[0..bufN], numN, numK) catch {
        INFO.* = 1;
        return;
    };

    var prev_logL: f64 = undefined;
    var this_logL: f64 = undefined;

    wgmm.stepE(xvec, wvec, numN, mvec, svec, pvec, numK, resp, &prev_logL);
    wgmm.stepM(xvec, wvec, numN, mvec, svec, pvec, numK, resp);

    var iter: usize = 1;

    while (iter < IMAX) : (iter += 1) {
        wgmm.stepE(xvec, wvec, numN, mvec, svec, pvec, numK, resp, &this_logL);

        if (this_logL <= prev_logL) {
            INFO.* = 2;
            break;
        }

        if ((this_logL - prev_logL) < std.math.floatEps(f64)) {
            INFO.* = 3;
            break;
        }

        wgmm.stepM(xvec, wvec, numN, mvec, svec, pvec, numK, resp);
    } else {
        INFO.* = 0;
    }
}
