const testing = @import("std").testing;

pub inline fn pow2(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * x,
        else => @compileError(""),
    };
}

pub inline fn pow3(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * (x * x),
        else => @compileError(""),
    };
}

pub inline fn pow4(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => pow2(T, x * x),
        else => @compileError(""),
    };
}

pub inline fn pow5(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * pow4(T, x),
        else => @compileError(""),
    };
}

test "pow2, pow3, pow4, pow5" {
    try testing.expectEqual(4, pow2(i32, 2));
    try testing.expectEqual(8, pow3(i32, 2));
    try testing.expectEqual(16, pow4(i32, 2));
    try testing.expectEqual(32, pow5(i32, 2));
}

inline fn muladd(comptime T: type, a: T, x: T, b: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => a * x + b,
        else => @compileError(""),
    };
}

// Evaluate [a₀, a₁⋅x, a₂⋅x², a₃⋅x³] by
// evalPolyArray(T, 3, x, [a₀, a₁, a₂, a₃])
pub fn evalPolyArray(
    comptime T: type,
    comptime order: usize,
    x: T,
    p: *[order + 1]T,
) void {
    var tmp: T = x;
    for (p[1..]) |*ptr| {
        ptr.* *= tmp;
        tmp *= x;
    }
    return;
}

// Evaluate a₃⋅x³ + a₂⋅x² + a₁⋅x + a₀ by
// evalpoly(T, 3, x, [a₃, a₂, a₁, a₀])
pub fn evalPolySum(
    comptime T: type,
    comptime order: usize,
    x: T,
    p: [order + 1]T,
) T {
    var tmp: T = p[0];
    // TODO: There should be an upper limit to allow unrolling the for-loop.
    inline for (p[1..]) |val| tmp = muladd(T, tmp, x, val);
    return tmp;
}

test "evalPoly" {
    // coeff = [a₀, a₁, a₂, a₃]
    var coeff: [3 + 1]i32 = .{ 1, 3, 4, 2 };
    // coeff -> [a₀, a₁⋅x, a₂⋅x², a₃⋅x³]
    evalPolyArray(i32, 3, 5, &coeff);
    // a₀⋅y³ + a₁⋅x⋅y² + a₂⋅x²⋅y + a₃⋅x³
    try testing.expectEqual(2028, evalPolySum(i32, 3, 7, coeff));
}
