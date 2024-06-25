value: f64 = undefined,
deriv: [2]f64 = undefined, // dΓ/dσ, dΓ/dγ

const G0: comptime_float = 1.0;
const G1: comptime_float = 0.07842;
const G2: comptime_float = 4.47163;
const G3: comptime_float = 2.42843;
const G4: comptime_float = 2.69269;
const G5: comptime_float = 1.0;
const S0: comptime_float = 2.0 * @sqrt(2.0 * @log(2.0));

pub fn update(self: *@This(), GammaG: f64, GammaL: f64) void { // checked
    const dGds_factor: comptime_float = comptime S0 / 5.0; // factor of dΓ/dσ
    const dGdg_factor: comptime_float = comptime 2.0 / 5.0; // factor of dΓ/dγ

    var cGtot: [6]f64 = comptime .{ G0, G1, G2, G3, G4, G5 };
    var cdGds: [5]f64 = comptime .{ G1 * 1.0, G2 * 2.0, G3 * 3.0, G4 * 4.0, G5 * 5.0 };
    var cdGdg: [5]f64 = comptime .{ G0 * 5.0, G1 * 4.0, G2 * 3.0, G3 * 2.0, G4 * 1.0 };

    var temp: f64 = GammaG;
    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |*c_Gtot, *c_dGds, *c_dGdg| {
        c_Gtot.* *= temp;
        c_dGds.* *= temp;
        c_dGdg.* *= temp;
        temp *= GammaG;
    }
    cGtot[5] *= temp;

    var Gtot: f64 = cGtot[0];
    var dGds: f64 = cdGds[0];
    var dGdg: f64 = cdGdg[0];

    for (cGtot[1..5], cdGds[1..], cdGdg[1..]) |c_Gtot, c_dGds, c_dGdg| {
        Gtot = Gtot * GammaL + c_Gtot;
        dGds = dGds * GammaL + c_dGds;
        dGdg = dGdg * GammaL + c_dGdg;
    }
    Gtot = Gtot * GammaL + cGtot[5];

    temp = @log(Gtot);
    Gtot = @exp(0.2 * temp);
    temp = @exp(-0.8 * temp);

    dGds = dGds_factor * temp * dGds;
    dGdg = dGdg_factor * temp * dGdg;

    self.value = Gtot;
    self.deriv[0] = dGds;
    self.deriv[1] = dGdg;

    return;
}
