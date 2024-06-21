//! Reference
//! William H. Press, Saul A. Teukolsky, William T. Vetemperling, Brian P. Flannery 2007
//! Numerical Recipes 3rd Edition: The Art of Scientific Computing, Sec. 9.4
const std = @import("std");

const NewtonError = error{
    OutOfBounds,
    MaxIteration,
};

/// Using a combination of Newton-Raphson and bisection, return the root of a fcalltion
/// bracketed between `xleft` and `xright`. The root will be refined until its accuracy
/// is known within `±xacc`.
pub fn findRoot(
    fcall: *const fn (x: f64) f64,
    deriv: *const fn (x: f64) f64,
    target: f64,
    xleft: f64,
    xright: f64,
    ITMAX: usize, // Maximum allowed number of iterations.
) NewtonError!f64 {
    const XACC: comptime_float = std.math.floatEps(f64);

    const f_lo: f64 = fcall(xleft) - target;
    const f_hi: f64 = fcall(xright) - target;

    // @panic("Root must be bracketed in `findRoot`.\n");
    if ((f_lo > 0.0 and f_hi > 0.0) or (f_lo < 0.0 and f_hi < 0.0)) {
        return NewtonError.OutOfBounds;
    }

    if (f_lo == 0.0) return xleft;
    if (f_hi == 0.0) return xright;

    var x_lo: f64 = undefined;
    var x_hi: f64 = undefined;

    if (f_lo < 0.0) {
        x_lo = xleft;
        x_hi = xright;
    } else {
        x_lo = xright;
        x_hi = xleft;
    }

    var root: f64 = 0.5 * (xleft + xright); // init. the guess for root
    var fval: f64 = fcall(root) - target;
    var grad: f64 = deriv(root);

    var last_step: f64 = @abs(xright - xleft); // the stepsize before last
    var this_step: f64 = last_step; // the last step.

    var temp: f64 = undefined;
    var flag: bool = undefined;

    for (0..ITMAX) |_| { // Loop over allowed iterations.
        // Use bisection if Newton is out of range, or not decreasing fast enough.
        flag = @abs(2.0 * fval) > @abs(last_step * grad); // if the step is too small

        if (!flag) {
            flag = ((root - x_hi) * grad - fval) * ((root - x_lo) * grad - fval) > 0.0;
        }

        if (flag) {
            last_step = this_step;
            this_step = 0.5 * (x_hi - x_lo);

            root = x_lo + this_step;

            if (x_lo == root) return root;
        } else {
            last_step = this_step;
            this_step = fval / grad;

            temp = root;
            root = root - this_step;

            if (temp == root) return root;
        }

        if (@abs(this_step) < XACC) return root;

        fval = fcall(root) - target;
        grad = deriv(root);

        if (fval < 0.0) {
            x_lo = root;
        } else {
            x_hi = root;
        }
    }

    return NewtonError.MaxIteration;
}
