const std = @import("std");
const testing = std.testing;

const RootError = error{DomainError};

pub fn nthRoot(comptime T: type, comptime n: comptime_int, x: T) RootError!T {
    if (@typeInfo(T) != .Float) @compileError("");
    if (!(n > 0)) @compileError("");

    switch (n) {
        1 => return x,
        2 => return @sqrt(x),
        else => {
            const odd_or_even: comptime_int = n & 1;
            if (x == 0.0) return 0.0;
            if (x > 0.0) {
                return @exp(@log(x) / n);
            } else {
                switch (odd_or_even) { // x < 0.0
                    0 => return RootError.DomainError,
                    1 => return -@exp(@log(-x) / n),
                    else => unreachable,
                }
            }
        },
    }
}

test "nthRoot" {
    try testing.expectEqual(0x1.52305f394d9dep2, nthRoot(f64, 5, 4120.0));
}
