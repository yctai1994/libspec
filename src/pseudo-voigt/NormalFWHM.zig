scale: f64 = undefined, // σ
value: f64 = undefined, // FWHM = 2⋅√( 2⋅log(2) )⋅σ

const fwhm_factor: comptime_float = 2.0 * @sqrt(2.0 * @log(2.0));

const Self: type = @This();

pub fn init(allocator: mem.Allocator) !*Self {
    const self: *Self = try allocator.create(Self);
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
    self.value = fwhm_factor * scale;
    return;
}

test "test" {
    const this: Self = .{};
    try std.testing.expectEqual(2 * @sizeOf(f64), @sizeOf(@TypeOf(this)));
}

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
