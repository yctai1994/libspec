const std = @import("std");

const findRoot = @import("./newton.zig").findRoot;
const NormalDist = @import("./prob-dist/normal.zig").NormalDist;
const LorentzDist = @import("./prob-dist/lorentz.zig").LorentzDist;
const PseudoVoigt = @import("./pseudo-voigt/PseudoVoigt.zig");
const PseudoVoigtGamma = @import("./pseudo-voigt/PseudoVoigtGamma.zig");
const PseudoVoigtEta = @import("./pseudo-voigt/PseudoVoigtEta.zig");
const PseudoVoigtLorentz = @import("./pseudo-voigt/PseudoVoigtLorentz.zig");

export fn c_find_root(
    fcall: *const fn (x: f64) callconv(.C) f64,
    deriv: *const fn (x: f64) callconv(.C) f64,
    target: f64,
    xleft: f64,
    xright: f64,
    ITMAX: usize, // Maximum allowed number of iterations.
) f64 {
    return findRoot(
        .C,
        fcall,
        deriv,
        target,
        xleft,
        xright,
        ITMAX,
    ) catch std.math.nan(f64);
}

export fn c_make_normal_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mode: f64,
    scale: f64,
) void {
    const dist: NormalDist = .{ .mode = mode, .scale = scale };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}

export fn c_make_lorentz_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mode: f64,
    scale: f64,
) void {
    const dist: LorentzDist = .{ .mode = mode, .scale = scale };
    return dist.writeAll(pptr[0..dims], xptr[0..dims]);
}

export fn c_make_pvoigt_dist(
    pptr: [*]f64,
    xptr: [*]f64,
    dims: usize,
    mode: f64,
    scaleN: f64,
    scaleL: f64,
) void {
    const page = std.heap.page_allocator;

    const pseudo_voigt: *PseudoVoigt = PseudoVoigt.init(page) catch {
        std.debug.print("fail, do nothing and return.\n", .{});
        return;
    };
    defer pseudo_voigt.deinit(page);

    pseudo_voigt.forward(mode, scaleN, scaleL);
    pseudo_voigt.writeAll(pptr[0..dims], xptr[0..dims]);
    return;
}

export fn c_dGamma(scaleN: f64, scaleL: f64, Gtot: *f64, deriv: [*]f64) void {
    const page = std.heap.page_allocator;

    const pseudo_voigt_gamma: *PseudoVoigtGamma = PseudoVoigtGamma.init(page) catch {
        std.debug.print("fail, do nothing and return.\n", .{});
        return;
    };
    defer pseudo_voigt_gamma.deinit(page);

    pseudo_voigt_gamma.forward(scaleN, scaleL);
    Gtot.* = pseudo_voigt_gamma.value;
    pseudo_voigt_gamma.backward();
    // deriv = [ dΓ/dμ, dΓ/dσ, dΓ/dγ ]
    @memcpy(deriv[1..3], &pseudo_voigt_gamma.deriv);

    return;
}

export fn c_dEta(scaleN: f64, scaleL: f64, Etot: *f64, deriv: [*]f64) void {
    const page = std.heap.page_allocator;

    const pseudo_voigt_eta: *PseudoVoigtEta = PseudoVoigtEta.init(page) catch {
        std.debug.print("fail, do nothing and return.\n", .{});
        return;
    };
    defer pseudo_voigt_eta.deinit(page);

    pseudo_voigt_eta.forward(scaleN, scaleL);
    Etot.* = pseudo_voigt_eta.value;
    pseudo_voigt_eta.backward();
    // deriv = [ dη/dμ, dη/dσ, dη/dγ ]
    @memcpy(deriv[1..3], &pseudo_voigt_eta.deriv);

    return;
}

export fn c_dLorentz(
    x: f64,
    mode: f64,
    scaleN: f64,
    scaleL: f64,
    Lprob: *f64,
    deriv: [*]f64,
) void {
    const page = std.heap.page_allocator;

    const pseudo_voigt: *PseudoVoigt = PseudoVoigt.init(page) catch {
        std.debug.print("fail, do nothing and return.\n", .{});
        return;
    };
    defer pseudo_voigt.deinit(page);

    pseudo_voigt.forward(mode, scaleN, scaleL);
    pseudo_voigt.eta.backward();
    pseudo_voigt.lorentz.backward(x, deriv[0..3]);

    Lprob.* = pseudo_voigt.lorentz.value; // deriv = [ dL/dμ, dL/dσ, dL/dγ ]

    return;
}
