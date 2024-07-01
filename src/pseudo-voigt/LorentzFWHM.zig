scale: f64 = undefined, // γ
value: f64 = undefined, // FWHM = 2γ
deriv: f64 = undefined, // dΓ/dγ

const Self: type = @This();

pub fn init(allocator: mem.Allocator) !*Self {
    const self: *Self = try allocator.create(Self);
    self.deriv = 2.0;
    return self;
}

pub fn deinit(self: *Self, allocator: mem.Allocator) void {
    allocator.destroy(self);
    return;
}

test "allocation" {
    const page = testing.allocator;
    const this: *Self = try Self.init(page);
    defer this.deinit(page);
}

pub fn forward(self: *Self, scale: f64) void {
    self.scale = scale;
    self.value = self.deriv * scale;
    return;
}

pub fn backward(self: *Self) f64 {
    return self.deriv; // dΓ/dγ
}

test "test" {
    const this: Self = .{};
    try std.testing.expectEqual(2 * @sizeOf(f64), @sizeOf(@TypeOf(this)));
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
