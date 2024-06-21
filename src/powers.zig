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
