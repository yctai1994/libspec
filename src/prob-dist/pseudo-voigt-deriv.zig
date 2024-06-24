const std = @import("std");
const math = std.math;

const poly = @import("../poly.zig");
const normal = @import("./normal.zig");
const lorentz = @import("./lorentz.zig");
const nthRoot = @import("../nth-root.zig").nthRoot;

const Gamma_0: comptime_float = 1.0;
const Gamma_1: comptime_float = 0.07842;
const Gamma_2: comptime_float = 4.47163;
const Gamma_3: comptime_float = 2.42843;
const Gamma_4: comptime_float = 2.69269;
const Gamma_5: comptime_float = 1.0;
const sigma_0: comptime_float = 2.0 * @sqrt(2.0 * @log(2.0));

pub fn Gamma(Gamma_G: f64, Gamma_L: f64) f64 { // checked
    var coeff: [5 + 1]f64 = comptime .{
        Gamma_0,
        Gamma_1,
        Gamma_2,
        Gamma_3,
        Gamma_4,
        Gamma_5,
    };
    poly.evalPolyArray(f64, 5, Gamma_G, &coeff);
    return nthRoot(f64, 5, poly.evalPolySum(f64, 5, Gamma_L, coeff)) catch unreachable;
}

pub fn dGamma(Gamma_G: f64, Gamma_L: f64, Gtot: *f64, dGds: *f64, dGdg: *f64) void { // checked
    const dGds_factor: comptime_float = comptime sigma_0 / 5.0; // d Gamma/d sigma
    const dGdg_factor: comptime_float = comptime 2.0 / 5.0; // d Gamma/d gamma

    var cGtot: [5 + 1]f64 = comptime .{
        Gamma_0,
        Gamma_1,
        Gamma_2,
        Gamma_3,
        Gamma_4,
        Gamma_5,
    };

    var cdGds: [4 + 1]f64 = comptime .{
        Gamma_1 * 1.0,
        Gamma_2 * 2.0,
        Gamma_3 * 3.0,
        Gamma_4 * 4.0,
        Gamma_5 * 5.0,
    };

    var cdGdg: [4 + 1]f64 = comptime .{
        Gamma_0 * 5.0,
        Gamma_1 * 4.0,
        Gamma_2 * 3.0,
        Gamma_3 * 2.0,
        Gamma_4 * 1.0,
    };

    var temp: f64 = Gamma_G;
    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |*c_Gtot, *c_dGds, *c_dGdg| {
        const tmp = @as(@Vector(3, f64), @splat(temp));
        const vec = @Vector(3, f64){ c_Gtot.*, c_dGds.*, c_dGdg.* } * tmp;
        inline for (.{ c_Gtot, c_dGds, c_dGdg }, 0..) |p, i| p.* = vec[i];
        temp *= Gamma_G;
    }
    cGtot[5] *= temp;

    Gtot.* = cGtot[0];
    dGds.* = cdGds[0];
    dGdg.* = cdGdg[0];
    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |c_Gtot, c_dGds, c_dGdg| {
        const x = @as(@Vector(3, f64), @splat(Gamma_L));
        const a = @Vector(3, f64){ Gtot.*, dGds.*, dGdg.* };
        const b = @Vector(3, f64){ c_Gtot, c_dGds, c_dGdg };
        const y = a * x + b;
        inline for (.{ Gtot, dGds, dGdg }, 0..) |p, i| p.* = y[i];
    }
    Gtot.* = Gtot.* * Gamma_L + cGtot[5];

    temp = @log(Gtot.*);
    Gtot.* = @exp(0.2 * temp);
    temp = @exp(-0.8 * temp);

    dGds.* = dGds_factor * temp * dGds.*;
    dGdg.* = dGdg_factor * temp * dGdg.*;

    return;
}
