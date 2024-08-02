const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    const dylib = b.addSharedLibrary(.{
        .name = "spec",
        .root_source_file = b.path("src/dylib.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });

    const install = b.addInstallArtifact(dylib, .{});
    b.default_step.dependOn(&install.step);

    const dylib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/dylib.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const run_dylib_unit_tests = b.addRunArtifact(dylib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_dylib_unit_tests.step);

    return;
}
