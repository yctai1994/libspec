const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});

    const lib = b.addSharedLibrary(.{
        .name = "spec",
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });

    const install = b.addInstallArtifact(lib, .{});
    b.default_step.dependOn(&install.step);

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    return;
}
