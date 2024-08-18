const Error = error{DimensionMismatch};

pub fn trmv(comptime uplo: u8, comptime trans: u8, A: [][]f64, x: []f64) Error!void {
    const n: usize = A.len;
    if (n != x.len) return Error.DimensionMismatch;

    switch (uplo) {
        'R' => {
            switch (trans) {
                'N' => {
                    // x ← R⋅x
                    var t: f64 = undefined;
                    for (0..n) |i| {
                        t = 0.0;
                        for (i..n) |j| t += A[i][j] * x[j];
                        x[i] = t;
                    }
                },
                'T' => {
                    // x ← Rᵀ⋅x
                    var t: f64 = undefined;
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        t = x[i];
                        x[i] = 0.0;
                        for (i..n) |j| x[j] += A[i][j] * t;
                        if (i == 0) break;
                    }
                },
                else => @compileError(""),
            }
        },
        // 'L' => {},
        else => @compileError(""),
    }
}

test "trmv: x ← R⋅x" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const R: [][]f64 = try ArrF64.matrix(4, 4);
    defer ArrF64.free(R);

    const x: []f64 = try ArrF64.vector(4);
    defer ArrF64.free(x);

    inline for (.{ 2.0, 6.0, 8.0, 1.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 7.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 4.0, 9.0 }, R[2]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 0.0, 3.0 }, R[3]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.5, 0.7, 0.2, 0.3 }, x) |val, *ptr| ptr.* = val;

    const z: [4]f64 = .{ 7.1, 3.6, 3.5, 0.9 };

    try trmv('R', 'N', R, x);

    for (x, &z, 0..) |x_i, z_i, i| {
        if (testing.expectApproxEqRel(x_i, z_i, 1e-15)) |_| {} else |_| {
            std.debug.print("x[{d}] = {d}, z[{d}] = {d}\n", .{ i, x_i, i, z_i });
        }
    }
}

test "trmv: x ← Rᵀ⋅x" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const R: [][]f64 = try ArrF64.matrix(4, 4);
    defer ArrF64.free(R);

    const x: []f64 = try ArrF64.vector(4);
    defer ArrF64.free(x);

    inline for (.{ 2.0, 6.0, 8.0, 1.0 }, R[0]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 1.0, 7.0, 5.0 }, R[1]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 4.0, 9.0 }, R[2]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.0, 0.0, 0.0, 3.0 }, R[3]) |val, *ptr| ptr.* = val;
    inline for (.{ 0.5, 0.7, 0.2, 0.3 }, x) |val, *ptr| ptr.* = val;

    const z: [4]f64 = .{ 1.0, 3.7, 9.7, 6.7 };

    try trmv('R', 'T', R, x);

    for (x, &z, 0..) |x_i, z_i, i| {
        if (testing.expectApproxEqRel(x_i, z_i, 1e-16)) |_| {} else |_| {
            std.debug.print("x[{d}] = {d}, z[{d}] = {d}\n", .{ i, x_i, i, z_i });
        }
    }
}

const std = @import("std");
const testing = std.testing;
const Array = @import("./array.zig").Array;
