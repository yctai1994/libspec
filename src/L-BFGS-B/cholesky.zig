const Errors = error{NotPositiveDefinite};

pub fn cholesky(A: [][]f64) !void {
    const n: usize = A.len;
    var t: f64 = undefined;

    for (0..n) |i| {
        for (i..n) |j| {
            t = A[i][j];
            for (0..i) |k| t -= A[i][k] * A[j][k];
            if (i == j) {
                if (t <= 0.0) return Errors.NotPositiveDefinite;
                A[i][i] = @sqrt(t);
            } else A[j][i] = t / A[i][i];
        }
    }

    for (0..n) |j| {
        for (0..j) |i| A[i][j] = 0.0;
    }
}
