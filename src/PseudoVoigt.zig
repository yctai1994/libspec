//! Tape:
//! [ dy/d(log-likelihood), dy/dPpV, dy/dPN,   dy/dPL, dy/dσV,
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

    self.mode = try PseudoVoigtMode.init(allocator, self.tape);
    errdefer self.mode.deinit(allocator);

    self.normal = try PseudoVoigtNormal.init(allocator, self.mode, self.gamma, self.tape);
    errdefer self.normal.deinit(allocator);

    self.lorentz = try PseudoVoigtLorentz.init(allocator, self.mode, self.gamma, self.tape);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    @memset(self.tape, 1.0);

    self.deriv_in = &self.tape[0]; // dy/d(log-likelihood)
    self.deriv_out = &self.tape[1]; // dy/d(PpV)

    return self;
}

fn deinit(self: *Self, allocator: mem.Allocator) void {
    self.lorentz.deinit(allocator);
    self.normal.deinit(allocator);
    self.mode.deinit(allocator);

    self.eta.deinit(allocator);
    self.gamma.deinit(allocator);

    allocator.free(self.tape);
    allocator.destroy(self);

    return;
}

fn preforward(self: *Self, mu: f64, sigma: f64, gamma: f64) void {
    self.mode.forward(mu);
    self.gamma.forward(sigma, gamma);
    self.normal.scale.forward();
    self.lorentz.scale.forward();
    self.eta.forward();
    return;
}

fn forward(self: *Self, x: f64) void {
    self.normal.forward(x);
    self.lorentz.forward(x);

    const arg1: f64 = self.eta.value;
    const arg2: f64 = 1.0 - arg1;

    self.value = arg1 * self.lorentz.value + arg2 * self.normal.value;

    self.normal.deriv = arg2; // dPpV/dPN
    self.lorentz.deriv = arg1; // dPpV/dPL
    self.eta.deriv = self.lorentz.value - self.normal.value; // dPpV/dη

    return;
}

fn backward(self: *Self, final_deriv_out: []f64) void {
    if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

    // (dy/dPpV) = (d(log-likelihood)/dPpV) × (dy/d(log-likelihood))
    self.deriv_out.* = self.deriv * self.deriv_in.*;

    self.normal.backward();
    self.lorentz.backward();
    self.mode.backward(final_deriv_out);

    self.eta.backward();
    self.gamma.backward(final_deriv_out);

    return;
}

test "Pseudo-Voigt Function Reverse Autodifferentiation, y = PpV" {
    const page = testing.allocator;

    const pseudo_voigt: *Self = try Self.init(page);
    defer pseudo_voigt.deinit(page);

    const deriv: []f64 = try page.alloc(f64, 3);
    defer page.free(deriv);

    pseudo_voigt.preforward(_test_mode_, _test_sigma_, _test_gamma_);

    // Produce dy/dPpV = dy/dPpV = 1
    pseudo_voigt.deriv = 1.0;

    pseudo_voigt.forward(_test_x_);
    pseudo_voigt.backward(deriv);

    std.debug.print(
        "PpV  = {d} @ (x = {d}, μ = {d}, σ = {d}, γ = {d})\n",
        .{ pseudo_voigt.value, _test_x_, _test_mode_, _test_sigma_, _test_gamma_ },
    );
    std.debug.print(
        "dPpV = {d} @ (x = {d}, μ = {d}, σ = {d}, γ = {d})\n",
        .{ deriv, _test_x_, _test_mode_, _test_sigma_, _test_gamma_ },
    );
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtEta = struct {
    value: f64 = undefined, // η
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

        self.deriv_in = &tape[1]; // dy/dPpV
        self.deriv_out = &tape[6]; // dy/dη

        return self;
    }

    inline fn deinit(self: *PseudoVoigtEta, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoVoigtEta) void {
        const alpha: f64 = self.lorentz.value / self.gamma.value; // α = ΓL/Γtot

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

    inline fn backward(self: *PseudoVoigtEta) void {
        // (dy/dη) = (dPpV/dη) × (dy/dPpV)
        self.deriv_out.* = self.deriv * self.deriv_in.*;
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtNormal = struct {
    value: f64 = undefined, // N( x | μ, Γ(σ, γ) )
    deriv: f64 = undefined, // dPpV/dPN
    deriv_in: *f64 = undefined, // dy/dPpV
    deriv_out: *f64 = undefined, // dy/dPN

    mode: *PseudoVoigtMode, // handled by PseudoVoigt
    scale: *PseudoNormalScale,

    fn init(
        allocator: mem.Allocator,
        mode: *PseudoVoigtMode,
        gamma: *PseudoVoigtGamma,
        tape: []f64,
    ) !*PseudoVoigtNormal {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoVoigtNormal);
        errdefer allocator.destroy(self);

        self.scale = try PseudoNormalScale.init(allocator, gamma, tape);
        self.mode = mode;

        self.deriv_in = &tape[1]; // dy/dPpV
        self.deriv_out = &tape[2]; // dy/dPN

        return self;
    }

    inline fn deinit(self: *PseudoVoigtNormal, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoVoigtNormal, x: f64) void {
        // self.scale.forward();

        const prob: f64 = PseudoVoigtNormal.density(x, self.mode.value, self.scale.value);
        const arg1: f64 = (x - self.mode.value) / pow2(self.scale.value);
        const arg2: f64 = self.scale.value * pow2(arg1) - 1.0 / self.scale.value;

        self.value = prob;
        self.mode.deriv[0] = prob * arg1; // [ dPN/dμ, dPL/dμ ]
        self.scale.deriv = prob * arg2; // dPN/dσV

        return;
    }

    inline fn backward(self: *PseudoVoigtNormal) void {
        // (dy/dPN) = (dPpV/dPN) × (dy/dPpV)
        self.deriv_out.* = self.deriv * self.deriv_in.*;
        self.scale.backward();

        return;
    }

    inline fn density(x: f64, mu: f64, gamma: f64) f64 {
        const temp: comptime_float = comptime 1.0 / @sqrt(2.0 * math.pi);
        return temp * @exp(-0.5 * pow2((x - mu) / gamma)) / gamma;
    }
};

const PseudoNormalScale = struct {
    value: f64 = undefined, // σV
    deriv: f64 = undefined, // dPN/dσV
    deriv_in: *f64 = undefined, // dy/dPN
    deriv_out: *f64 = undefined, // dy/dσV

    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt

    fn init(
        allocator: mem.Allocator,
        gamma: *PseudoVoigtGamma,
        tape: []f64,
    ) !*PseudoNormalScale {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoNormalScale);
        self.gamma = gamma;

        self.deriv_in = &tape[2]; // dy/dPN
        self.deriv_out = &tape[4]; // dy/dσV

        return self;
    }

    inline fn deinit(self: *PseudoNormalScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoNormalScale) void {
        // Gamma should be already forwarded by `PseudoVoigt.preforward()`.
        const temp: comptime_float = comptime 0.5 / @sqrt(2.0 * @log(2.0));
        self.gamma.deriv[0] = temp; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
        self.value = temp * self.gamma.value;

        return;
    }

    inline fn backward(self: *PseudoNormalScale) void {
        // (dy/dσV) = (dPN/dσV) × (dy/dPN)
        self.deriv_out.* = self.deriv * self.deriv_in.*;

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtLorentz = struct {
    value: f64 = undefined, // L( x | μ, Γ(σ, γ) )
    deriv: f64 = undefined, // dPpV/dPL
    deriv_in: *f64 = undefined, // dy/dPpV
    deriv_out: *f64 = undefined, // dy/dPL

    mode: *PseudoVoigtMode, // handled by PseudoVoigt
    scale: *PseudoLorentzScale,

    fn init(
        allocator: mem.Allocator,
        mode: *PseudoVoigtMode,
        gamma: *PseudoVoigtGamma,
        tape: []f64,
    ) !*PseudoVoigtLorentz {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoVoigtLorentz);
        errdefer allocator.destroy(self);

        self.scale = try PseudoLorentzScale.init(allocator, gamma, tape);
        self.mode = mode;

        self.deriv_in = &tape[1]; // dy/dPpV
        self.deriv_out = &tape[3]; // dy/dPL

        return self;
    }

    inline fn deinit(self: *PseudoVoigtLorentz, allocator: mem.Allocator) void {
        self.scale.deinit(allocator);
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoVoigtLorentz, x: f64) void {
        // self.scale.forward();

        const twopi: comptime_float = comptime 2.0 * math.pi;
        const prob: f64 = PseudoVoigtLorentz.density(x, self.mode.value, self.scale.value);
        const arg1: f64 = prob / self.scale.value;
        const arg2: f64 = twopi * pow2(prob);

        self.value = prob;
        self.mode.deriv[1] = (x - self.mode.value) * arg2 / self.scale.value; // [ dPN/dμ, dPL/dμ ]
        self.scale.deriv = arg1 - arg2; // dPL/dγV

        return;
    }

    inline fn backward(self: *PseudoVoigtLorentz) void {
        // (dy/dPL) = (dPpV/dPL) × (dy/dPpV)
        self.deriv_out.* = self.deriv * self.deriv_in.*;
        self.scale.backward();

        return;
    }

    inline fn density(x: f64, mu: f64, gamma: f64) f64 {
        const temp: comptime_float = comptime 1.0 / math.pi;
        return temp / (gamma * (1.0 + pow2((x - mu) / gamma)));
    }
};

const PseudoLorentzScale = struct {
    value: f64 = undefined, // γV
    deriv: f64 = undefined, // dPL/dγV
    deriv_in: *f64 = undefined, // dy/dPL
    deriv_out: *f64 = undefined, // dy/dγV

    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt

    fn init(
        allocator: mem.Allocator,
        gamma: *PseudoVoigtGamma,
        tape: []f64,
    ) !*PseudoLorentzScale {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoLorentzScale);
        self.gamma = gamma;

        self.deriv_in = &tape[3]; // dy/dPL
        self.deriv_out = &tape[5]; // dy/dγV

        return self;
    }

    inline fn deinit(self: *PseudoLorentzScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    fn forward(self: *PseudoLorentzScale) void {
        // Gamma should be already forwarded by PseudoVoigt.
        self.gamma.deriv[1] = 0.5; // [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]
        self.value = 0.5 * self.gamma.value;

        return;
    }

    inline fn backward(self: *PseudoLorentzScale) void {
        // (dy/dγV) = (dPL/dγV) × (dy/dPL)
        self.deriv_out.* = self.deriv * self.deriv_in.*;

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtGamma = struct {
    value: f64 = undefined, // Γtot
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

        self.deriv_in = tape[4..7]; // [ dy/dσV, dy/dγV, dy/dη ]
        self.deriv_out = &tape[7]; // dy/dΓtot

        return self;
    }

    inline fn deinit(self: *PseudoVoigtGamma, allocator: mem.Allocator) void {
        self.lorentz.deinit(allocator);
        self.normal.deinit(allocator);
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
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

        // (dy/dΓtot) = [ dσV/dΓtot, dγV/dΓtot, dη/dΓtot ]ᵀ ⋅ [ dy/dσV, dy/dγV, dy/dη ]
        var temp: f64 = 0.0;
        for (self.deriv, self.deriv_in) |deriv, deriv_in| temp += deriv * deriv_in;

        self.deriv_out.* = temp;

        self.normal.backward(final_deriv_out);
        self.lorentz.backward(final_deriv_out);

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const NormalFWHM = struct {
    value: f64 = undefined, // ΓG
    deriv: f64 = undefined, // dΓtot/dΓG
    deriv_in: *f64 = undefined, // dy/dΓtot
    deriv_out: *f64 = undefined, // dy/dΓG

    scale: *NormalScale,

    fn init(allocator: mem.Allocator, tape: []f64) !*NormalFWHM {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(NormalFWHM);
        errdefer allocator.destroy(self);

        self.scale = try NormalScale.init(allocator, tape);

        self.deriv_in = &tape[7]; // dy/dΓtot
        self.deriv_out = &tape[8]; // dy/dΓG

        return self;
    }

    inline fn deinit(self: *NormalFWHM, allocator: mem.Allocator) void {
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
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

        // dy/dΓG = (dΓtot/dΓG) × (dy/dΓtot)
        self.deriv_out.* = self.deriv * self.deriv_in.*;

        self.scale.backward(final_deriv_out);

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const LorentzFWHM = struct {
    value: f64 = undefined, // ΓL
    deriv: [2]f64 = undefined, // [ dη/dΓL, dΓtot/dΓL ]
    deriv_in: []f64 = undefined, // [ dy/dη, dy/dΓtot ]
    deriv_out: *f64 = undefined, // dy/dΓL

    scale: *LorentzScale,

    fn init(allocator: mem.Allocator, tape: []f64) !*LorentzFWHM {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(LorentzFWHM);
        errdefer allocator.destroy(self);

        self.scale = try LorentzScale.init(allocator, tape);

        self.deriv_in = tape[6..8]; // [ dy/dη, dy/dΓtot ]
        self.deriv_out = &tape[9]; // dy/dΓL

        return self;
    }

    inline fn deinit(self: *LorentzFWHM, allocator: mem.Allocator) void {
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
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

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
    value: f64 = undefined, // μ
    deriv: [2]f64 = undefined, // [ dPN/dμ, dPL/dμ ]
    deriv_in: []f64 = undefined, // [ dy/dPN, dy/dPL ]

    fn init(allocator: mem.Allocator, tape: []f64) !*PseudoVoigtMode {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(PseudoVoigtMode);

        self.deriv_in = tape[2..4]; // [ dy/dPN, dy/dPL ]

        return self;
    }

    inline fn deinit(self: *PseudoVoigtMode, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    inline fn forward(self: *PseudoVoigtMode, mode: f64) void {
        self.value = mode;
        return;
    }

    fn backward(self: *PseudoVoigtMode, final_deriv_out: []f64) void {
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

        // (dy/dμ) = [ dPN/dμ, dPL/dμ ]ᵀ ⋅ [ dy/dPN, dy/dPL ]
        var temp: f64 = 0.0;
        for (self.deriv, self.deriv_in) |deriv, deriv_in| temp += deriv * deriv_in;

        final_deriv_out[0] = temp;

        return;
    }
};

const NormalScale = struct {
    value: f64 = undefined, // σ
    deriv: f64 = undefined, // dΓG/dσ
    deriv_in: *f64 = undefined, // dy/dΓG

    fn init(allocator: mem.Allocator, tape: []f64) !*NormalScale {
        if (tape.len != 10) unreachable;

        const self = try allocator.create(NormalScale);

        self.deriv_in = &tape[8]; // dy/dΓG

        return self;
    }

    inline fn deinit(self: *NormalScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    inline fn forward(self: *NormalScale, scale: f64) void {
        self.value = scale;
        return;
    }

    fn backward(self: *NormalScale, final_deriv_out: []f64) void {
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

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

        self.deriv_in = &tape[9]; // dy/dΓL

        return self;
    }

    inline fn deinit(self: *LorentzScale, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }

    inline fn forward(self: *LorentzScale, scale: f64) void {
        self.value = scale;
        return;
    }

    fn backward(self: *LorentzScale, final_deriv_out: []f64) void {
        if (final_deriv_out.len != 3) unreachable; // [ dy/dμ, dy/dσ, dy/dγ ]

        // dy/dγ = (dΓL/dγ) × (dy/dΓL)
        final_deriv_out[2] = self.deriv * self.deriv_in.*;

        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

fn pow2(x: f64) f64 {
    return x * x;
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const _test_x_: comptime_float = 1.213;
const _test_mode_: comptime_float = 0.878;
const _test_sigma_: comptime_float = 2.171;
const _test_gamma_: comptime_float = 1.305;

const std = @import("std");
const mem = std.mem;
const math = std.math;
const debug = std.debug;
const testing = std.testing;
