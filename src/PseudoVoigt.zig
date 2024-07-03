//! Tape:
//! [ dy/d(log-likelihood), dy/dPpV, dy/dPG,   dy/dPL, dy/dσV,
//!   dy/dγV,               dy/dη,   dy/dΓtot, dy/dΓG, dy/dΓL, ]
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

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.gamma = try PseudoVoigtGamma.init(allocator);
    errdefer self.gamma.deinit(allocator);

    self.eta = try PseudoVoigtEta.init(allocator, self.gamma);
    errdefer self.eta.deinit(allocator);

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    self.mode = try PseudoVoigtMode.init(allocator);

    self.normal = try PseudoVoigtNormal.init(
        allocator,
        self.mode,
        self.gamma,
    );
    errdefer self.normal.deinit(allocator);

    self.lorentz = try PseudoVoigtLorentz.init(
        allocator,
        self.mode,
        self.gamma,
    );

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

    var pseudo_voigt: *Self = try Self.init(page);
    defer pseudo_voigt.deinit(page);
    _ = &pseudo_voigt;

    debug.print("{any}\n", .{pseudo_voigt.*});
    debug.print("@sizeOf(PseudoVoigt) = {d}\n", .{@sizeOf(Self)});
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const PseudoVoigtEta = struct {
    gamma: *PseudoVoigtGamma, // handled by PseudoVoigt
    lorentz: *LorentzFWHM, // handled by PseudoVoigtGamma

    fn init(allocator: mem.Allocator, gamma: *PseudoVoigtGamma) !*PseudoVoigtEta {
        const self = try allocator.create(PseudoVoigtEta);
        self.gamma = gamma;
        self.lorentz = gamma.lorentz;
        return self;
    }

    fn deinit(self: *PseudoVoigtEta, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }
};

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
    normal: *NormalFWHM,
    lorentz: *LorentzFWHM,

    fn init(allocator: mem.Allocator) !*PseudoVoigtGamma {
        const self = try allocator.create(PseudoVoigtGamma);
        errdefer allocator.destroy(self);

        self.normal = try NormalFWHM.init(allocator);
        errdefer self.normal.deinit(allocator);

        self.lorentz = try LorentzFWHM.init(allocator);

        return self;
    }

    fn deinit(self: *PseudoVoigtGamma, allocator: mem.Allocator) void {
        self.normal.deinit(allocator);
        self.lorentz.deinit(allocator);
        allocator.destroy(self);
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const NormalFWHM = struct {
    scale: *NormalScale,

    fn init(allocator: mem.Allocator) !*NormalFWHM {
        const self = try allocator.create(NormalFWHM);
        errdefer allocator.destroy(self);

        self.scale = try NormalScale.init(allocator);

        return self;
    }

    fn deinit(self: *NormalFWHM, allocator: mem.Allocator) void {
        allocator.destroy(self);
        return;
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const LorentzFWHM = struct {
    scale: *LorentzScale,

    fn init(allocator: mem.Allocator) !*LorentzFWHM {
        const self = try allocator.create(LorentzFWHM);
        errdefer allocator.destroy(self);

        self.scale = try LorentzScale.init(allocator);

        return self;
    }

    fn deinit(self: *LorentzFWHM, allocator: mem.Allocator) void {
        allocator.destroy(self);
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
    fn init(allocator: mem.Allocator) !*NormalScale {
        return try allocator.create(NormalScale);
    }
};

const LorentzScale = struct {
    fn init(allocator: mem.Allocator) !*LorentzScale {
        return try allocator.create(LorentzScale);
    }
};

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

const std = @import("std");
const mem = std.mem;
const math = std.math;
const debug = std.debug;
const testing = std.testing;
