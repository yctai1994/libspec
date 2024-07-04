//! Tape:
//! [ dy/d(log-likelihood), dy/dPpV, dy/dPG,   dy/dPL, dy/dσV,
//!                 dy/dγV, dy/dη,   dy/dΓtot, dy/dΓG, dy/dΓL ]
tape: []f64,
value: f64, // PpV
deriv: f64, // d(log-likelihood)/d(PpV)
deriv_in: *f64, // dy/d(log-likelihood)
deriv_out: *f64, // dy/d(PpV)

eta: *PseudoVoigtEta,
gamma: *PseudoVoigtGamma, // for convenience

mode: *PseudoVoigtMode, // for convenience
normal: *PseudoVoigtNormal,
lorentz: *PseudoVoigtLorentz,

const Self: type = @This(); // PseudoVoigt

fn init(allocator: mem.Allocator) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.tape = try allocator.alloc(f64, 10);
    errdefer allocator.free(self.tape);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.gamma = try PseudoVoigtGamma.init(allocator, self.tape);
    errdefer self.gamma.deinit(allocator);

    self.eta = try PseudoVoigtEta.init(allocator, self.gamma, self.tape);
    errdefer self.eta.deinit(allocator);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.mode = try PseudoVoigtMode.init(allocator);

    self.normal = try PseudoVoigtNormal.init(allocator, self.mode, self.gamma);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try PseudoVoigtLorentz.init(allocator, self.mode, self.gamma);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    @memset(self.tape, 1.0);

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.lorentz.deinit(allocator);
    self.normal.deinit(allocator);

    self.eta.deinit(allocator);
    self.gamma.deinit(allocator);

    allocator.free(self.tape);
    allocator.destroy(self);

    return;
}

test "init" {
    const page = testing.allocator;

    const pseudo_voigt: *Self = try Self.init(page);
    defer pseudo_voigt.deinit(page);

    debug.print("{any}\n", .{pseudo_voigt.*});
    debug.print("@sizeOf(PseudoVoigt) = {d}\n", .{@sizeOf(Self)});
    debug.print(
        "@sizeOf(PseudoVoigtGamma) = {d}\n",
        .{@sizeOf(@TypeOf(pseudo_voigt.gamma.*))},
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtEta = struct {
    value: f64 = undefined,
    deriv: f64 = undefined, // dPpV/dη
    deriv_in: *f64 = undefined, // dy/dPpV
    deriv_out: *f64 = undefined, // dy/dη

    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt
    lorentz: *LorentzFWHM, // handled by PseudoVoigtGamma

    const Eta0: comptime_float = 0.0; // η₀
    const Eta1: comptime_float = 1.36603; // η₁
    const Eta2: comptime_float = -0.47719; // η₂
    const Eta3: comptime_float = 0.11116; // η₃

    fn init(
        allocator: mem.Allocator,
        gamma: *PseudoVoigtGamma,
        tape: []f64,
    ) !*PseudoVoigtEta {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoVoigtEta);
        self.gamma = gamma;
        self.lorentz = gamma.lorentz;

        self.deriv_in = &tape[1];
        self.deriv_out = &tape[6];

        return self;
    }

    fn deinit(self: *PseudoVoigtEta, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoVoigtEta) void {
        const alpha: f64 = self.lorentz.value / self.gamma.value;

        var eta: f64 = Eta3; // η = η₀ + η₁α + η₂α² + η₃α³
        var beta: f64 = 0.0; // β = dη/dα = η₁ + 2η₂α + 3η₃α²

        inline for (.{ Eta2, Eta1, Eta0 }) |coeff| {
            beta = beta * alpha + eta;
            eta = eta * alpha + coeff;
        }

        self.value = eta;

        beta /= self.gamma.value;

        self.gamma.deriv[2] = -alpha * beta; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
        self.lorentz.deriv[0] = beta; // [ dη/dΓL, dΓtot/dΓL ]

        return;
    }

    fn backward(self: *PseudoVoigtEta, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // (dy/dη) = (dPpV/dη) × (dy/dPpV)
        self.deriv_out.* = self.deriv * self.deriv_in.*;

        return;
    }
};

test "backward: y ≡ η" {
    const page = testing.allocator;

    const pseudo_voigt: *Self = try Self.init(page);
    defer pseudo_voigt.deinit(page);

    const deriv: []f64 = try page.alloc(f64, 3);
    defer page.free(deriv);

    const _sigma_: f64 = 2.171;
    const _gamma_: f64 = 1.305;

    pseudo_voigt.gamma.forward(_sigma_, _gamma_);
    pseudo_voigt.eta.forward();

    // Produce dy/dη = dη/dη = 1
    pseudo_voigt.eta.deriv = 1.0;

    pseudo_voigt.eta.backward(deriv);
    pseudo_voigt.gamma.backward(deriv);

    std.debug.print(
        "Eta        = {d} @ ({d}, {d})\n",
        .{ pseudo_voigt.eta.value, _sigma_, _gamma_ },
    );
    std.debug.print(
        "dEta       = {d} @ ({d}, {d})\n",
        .{ deriv, _sigma_, _gamma_ },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtNormal = struct {
    mode: *PseudoVoigtMode, // handled by PseudoVoigt
    scale: *PseudoNormalScale,

    fn init(
        allocator: mem.Allocator,
        mode: *PseudoVoigtMode,
        gamma: *PseudoVoigtGamma,
    ) !*PseudoVoigtNormal {
        const self = try allocator.create(PseudoVoigtNormal);
        errdefer allocator.destroy(self);

        self.scale = try PseudoNormalScale.init(allocator, gamma);
        self.mode = mode;

        return self;
    }

    fn deinit(self: *PseudoVoigtNormal, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }
};

const PseudoNormalScale = struct {
    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt

    fn init(
        allocator: mem.Allocator,
        gamma: *PseudoVoigtGamma,
    ) !*PseudoNormalScale {
        const self = try allocator.create(PseudoNormalScale);
        self.gamma = gamma;
        return self;
    }

    fn deinit(self: *PseudoNormalScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtLorentz = struct {
    mode: *PseudoVoigtMode, // handled by PseudoVoigt
    scale: *PseudoLorentzScale,

    fn init(
        allocator: mem.Allocator,
        mode: *PseudoVoigtMode,
        gamma: *PseudoVoigtGamma,
    ) !*PseudoVoigtLorentz {
        const self = try allocator.create(PseudoVoigtLorentz);
        errdefer allocator.destroy(self);

        self.scale = try PseudoLorentzScale.init(allocator, gamma);
        self.mode = mode;

        return self;
    }

    fn deinit(self: *PseudoVoigtLorentz, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }
};

const PseudoLorentzScale = struct {
    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt

    fn init(
        allocator: mem.Allocator,
        gamma: *PseudoVoigtGamma,
    ) !*PseudoLorentzScale {
        const self = try allocator.create(PseudoLorentzScale);
        self.gamma = gamma;
        return self;
    }

    fn deinit(self: *PseudoLorentzScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtGamma = struct {
    value: f64 = undefined,
    deriv: [3]f64 = undefined, // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
    deriv_in: []f64 = undefined, // [ dy/dσV, dy/dγV, dy/dη ]
    deriv_out: *f64 = undefined, // dy/dΓtot

    normal: *NormalFWHM,
    lorentz: *LorentzFWHM,

    const G0: comptime_float = 1.0;
    const G1: comptime_float = 0.07842;
    const G2: comptime_float = 4.47163;
    const G3: comptime_float = 2.42843;
    const G4: comptime_float = 2.69269;
    const G5: comptime_float = 1.0;

    fn init(allocator: mem.Allocator, tape: []f64) !*PseudoVoigtGamma {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoVoigtGamma);
        errdefer allocator.destroy(self);

        self.normal = try NormalFWHM.init(allocator, tape);
        errdefer self.normal.deinit(allocator);

        self.lorentz = try LorentzFWHM.init(allocator, tape);

        self.deriv_in = tape[4..7];
        self.deriv_out = &tape[7];

        return self;
    }

    fn deinit(self: *PseudoVoigtGamma, allocator: mem.Allocator) void {
        self.normal.deinit(allocator);
        self.lorentz.deinit(allocator);
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoVoigtGamma, sigma: f64, gamma: f64) void {
        self.normal.forward(sigma);
        self.lorentz.forward(gamma);

        const GammaG: f64 = self.normal.value;
        const GammaL: f64 = self.lorentz.value;

        var cGtot: [6]f64 = comptime .{ G0, G1, G2, G3, G4, G5 };
        var cdGdG: [5]f64 = comptime .{ G1 * 1.0, G2 * 2.0, G3 * 3.0, G4 * 4.0, G5 * 5.0 };
        var cdGdL: [5]f64 = comptime .{ G0 * 5.0, G1 * 4.0, G2 * 3.0, G3 * 2.0, G4 * 1.0 };

        var temp: f64 = GammaG;
        for (cGtot[1..5], cdGdG[1..], cdGdL[1..]) |*c_Gtot, *c_dGdG, *c_dGdL| {
            c_Gtot.* *= temp;
            c_dGdG.* *= temp;
            c_dGdL.* *= temp;
            temp *= GammaG;
        }
        cGtot[5] *= temp;

        var Gtot: f64 = cGtot[0];
        var dGdG: f64 = cdGdG[0];
        var dGdL: f64 = cdGdL[0];

        for (cGtot[1..5], cdGdG[1..], cdGdL[1..]) |c_Gtot, c_dGdG, c_dGdL| {
            Gtot = Gtot * GammaL + c_Gtot;
            dGdG = dGdG * GammaL + c_dGdG;
            dGdL = dGdL * GammaL + c_dGdL;
        }
        Gtot = Gtot * GammaL + cGtot[5];

        temp = @log(Gtot);
        Gtot = @exp(0.2 * temp);
        temp = @exp(-0.8 * temp) * 0.2;

        self.value = Gtot;

        self.normal.deriv = temp * dGdG; // dΓtot/dΓG
        self.lorentz.deriv[1] = temp * dGdL; // [ dη/dΓL, dΓtot/dΓL ]

        return;
    }

    fn backward(self: *PseudoVoigtGamma, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // (dy/dΓtot) = [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]ᵀ ⋅ [ dy/dσV, dy/dγV, dy/dη ]
        var temp: f64 = 0.0;
        for (self.deriv, self.deriv_in) |deriv, deriv_in| temp += deriv * deriv_in;

        self.deriv_out.* = temp;

        self.normal.backward(final_deriv_out);
        self.lorentz.backward(final_deriv_out);

        return;
    }
};

test "backward: y ≡ Γtot" {
    const page = testing.allocator;

    const pseudo_voigt: *Self = try Self.init(page);
    defer pseudo_voigt.deinit(page);

    const deriv: []f64 = try page.alloc(f64, 3);
    defer page.free(deriv);

    const _sigma_: f64 = 2.171;
    const _gamma_: f64 = 1.305;

    pseudo_voigt.gamma.forward(_sigma_, _gamma_);

    // Produce dy/dΓtot = dΓtot/dΓtot = 1
    pseudo_voigt.gamma.deriv = .{
        0x1.5555555555555p-2,
        0x1.5555555555555p-2,
        0x1.5555555555555p-2,
    };

    pseudo_voigt.gamma.backward(deriv);

    std.debug.print(
        "Gamma_tot  = {d} @ ({d}, {d})\n",
        .{ pseudo_voigt.gamma.value, _sigma_, _gamma_ },
    );
    std.debug.print(
        "dGamma_tot = {d} @ ({d}, {d})\n",
        .{ deriv, _sigma_, _gamma_ },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const NormalFWHM = struct {
    value: f64 = undefined,
    deriv: f64 = undefined, // dΓtot/dΓG
    deriv_in: *f64 = undefined, // dy/dΓtot
    deriv_out: *f64 = undefined, // dy/dΓG

    scale: *NormalScale,

    fn init(allocator: mem.Allocator, tape: []f64) !*NormalFWHM {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(NormalFWHM);
        errdefer allocator.destroy(self);

        self.scale = try NormalScale.init(allocator, tape);

        self.deriv_in = &tape[7];
        self.deriv_out = &tape[8];

        return self;
    }

    fn deinit(self: *NormalFWHM, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }

    fn forward(self: *NormalFWHM, scale: f64) void {
        const temp: comptime_float = comptime 2.0 * @sqrt(2.0 * @log(2.0));
        self.scale.forward(scale);
        self.scale.deriv = temp; // dΓG/dσ
        self.value = temp * scale;
        return;
    }

    fn backward(self: *NormalFWHM, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // dy/dΓG = (dΓtot/dΓG) × (dy/dΓtot)
        self.deriv_out.* = self.deriv * self.deriv_in.*;

        self.scale.backward(final_deriv_out);

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const LorentzFWHM = struct {
    value: f64 = undefined,
    deriv: [2]f64 = undefined, // [ dη/dΓL, dΓtot/dΓL ]
    deriv_in: []f64 = undefined, // [ dy/dη, dy/dΓtot ]
    deriv_out: *f64 = undefined, // dy/dΓL

    scale: *LorentzScale,

    fn init(allocator: mem.Allocator, tape: []f64) !*LorentzFWHM {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(LorentzFWHM);
        errdefer allocator.destroy(self);

        self.scale = try LorentzScale.init(allocator, tape);

        self.deriv_in = tape[6..8];
        self.deriv_out = &tape[9];

        return self;
    }

    fn deinit(self: *LorentzFWHM, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }

    fn forward(self: *LorentzFWHM, scale: f64) void {
        self.scale.forward(scale);
        self.scale.deriv = 2.0; // dΓL/dγ
        self.value = 2.0 * scale;
        return;
    }

    fn backward(self: *LorentzFWHM, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // (dy/dΓL) = [ dη/dΓL, dΓtot/dΓL ]ᵀ ⋅ [ dy/dη, dy/dΓtot ]
        var temp: f64 = 0.0;
        for (self.deriv, self.deriv_in) |deriv, deriv_in| temp += deriv * deriv_in;

        self.deriv_out.* = temp;
        self.scale.backward(final_deriv_out);

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtMode = struct {
    fn init(allocator: mem.Allocator) !*PseudoVoigtMode {
        return try allocator.create(PseudoVoigtMode);
    }
};

const NormalScale = struct {
    value: f64 = undefined, // σ
    deriv: f64 = undefined, // dΓG/dσ
    deriv_in: *f64 = undefined, // dy/dΓG

    fn init(allocator: mem.Allocator, tape: []f64) !*NormalScale {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(NormalScale);
        self.deriv_in = &tape[8];
        return self;
    }

    fn deinit(self: *NormalScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *NormalScale, scale: f64) void {
        self.value = scale;
        return;
    }

    fn backward(self: *NormalScale, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // dy/dσ = (dΓG/dσ) × (dy/dΓG)
        final_deriv_out[1] = self.deriv * self.deriv_in.*;
        return;
    }
};

const LorentzScale = struct {
    value: f64 = undefined, // γ
    deriv: f64 = undefined, // dΓL/dγ
    deriv_in: *f64 = undefined, // dy/dΓL

    fn init(allocator: mem.Allocator, tape: []f64) !*LorentzScale {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(LorentzScale);
        self.deriv_in = &tape[9];
        return self;
    }

    fn deinit(self: *LorentzScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *LorentzScale, scale: f64) void {
        self.value = scale;
        return;
    }

    fn backward(self: *LorentzScale, final_deriv_out: []f64) void {
        // final_deriv_out := [ dy/dμ, dy/dσ, dy/dγ ]
        if (final_deriv_out.len != 3) unreachable;

        // dy/dγ = (dΓL/dγ) × (dy/dΓL)
        final_deriv_out[2] = self.deriv * self.deriv_in.*;
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const math = std.math;
const debug = std.debug;
const testing = std.testing;
