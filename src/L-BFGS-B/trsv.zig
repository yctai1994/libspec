const Error = error{DimensionMismatch};

pub fn trsv(comptime uplo: u8, comptime trans: u8, A: [][]f64, x: []f64) Error!void {
    const n: usize = A.len;
    if (n != x.len) return Error.DimensionMismatch;

    switch (uplo) {
        'R' => {
            switch (trans) {
                'N' => {
                    // R⋅x = b
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        for (i + 1..n) |j| x[i] -= A[i][j] * x[j];
                        x[i] /= A[i][i];
                        if (i == 0) break;
                    }
                },
                'T' => {
                    // Rᵀ⋅x = b
                    for (0..n) |i| {
                        x[i] /= A[i][i];
                        for (i + 1..n) |j| x[j] -= A[i][j] * x[i];
                    }
                },
                else => @compileError(""),
            }
        },
        'L' => {
            switch (trans) {
                'N' => {
                    // L⋅x = b
                    var t: f64 = undefined;
                    for (0..n) |i| {
                        t = x[i];
                        for (0..i) |j| t -= A[i][j] * x[j];
                        x[i] = t / A[i][i];
                    }
                },
                'T' => {
                    // Lᵀ⋅x = b
                    var i: usize = n - 1;
                    while (true) : (i -= 1) {
                        x[i] /= A[i][i];
                        for (0..i) |j| x[j] -= A[i][j] * x[i];
                        if (i == 0) break;
                    }
                },
                else => @compileError(""),
            }
        },
        else => @compileError(""),
    }
}
