const testing = @import("std").testing;

pub inline fn square(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * x,
        else => @compileError(""),
    };
}

pub inline fn cubic(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * (x * x),
        else => @compileError(""),
    };
}

pub inline fn quartic(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => square(T, x * x),
        else => @compileError(""),
    };
}

pub inline fn quintic(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .Int, .Float => x * quartic(T, x),
        else => @compileError(""),
    };
}

test "square, cubic, quartic, quintic" {
    try testing.expectEqual(4, square(i32, 2));
    try testing.expectEqual(8, cubic(i32, 2));
    try testing.expectEqual(16, quartic(i32, 2));
    try testing.expectEqual(32, quintic(i32, 2));
}
