//! Implements the Dao-Thaler + Gruen optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf

use allocative::Allocative;

use super::dense_mlpoly::DensePolynomial;
use super::multilinear_polynomial::BindingOrder;
use crate::{field::JoltField, poly::eq_poly::EqPolynomial};

#[derive(Debug, Clone, PartialEq, Allocative)]
/// A struct holding the equality polynomial evaluations for use in sum-check, when incorporating
/// both the Gruen and Dao-Thaler optimizations.
///
/// For the `i = 0..n`-th round of sum-check, we want the following invariants (low to high):
///
/// - `current_index = n - i` (where `n = w.len()`)
/// - `current_scalar = eq(w[(n - i)..],r[..i])`
/// - `E_out_vec.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E_in_vec.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E_in_vec` is empty
///
/// Implements both LowToHigh ordering and HighToLow ordering.
pub struct GruenSplitEqPolynomial<F: JoltField> {
    pub current_index: usize,
    pub current_scalar: F,
    pub w: Vec<F>,
    pub E_in_vec: Vec<Vec<F>>,
    pub E_out_vec: Vec<Vec<F>>,
    pub binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new")]
    pub fn new(w: &[F], binding_order: BindingOrder) -> Self {
        match binding_order {
            BindingOrder::LowToHigh => {
                let m = w.len() / 2;
                //   w = [w_out, w_in, w_last]
                //         ↑      ↑      ↑
                //         |      |      |
                //         |      |      last element
                //         |      second half of remaining elements (for E_in)
                //         first half of remaining elements (for E_out)
                let (_, wprime) = w.split_last().unwrap();
                let (w_out, w_in) = wprime.split_at(m);
                let (E_out_vec, E_in_vec) = rayon::join(
                    || EqPolynomial::evals_cached(w_out),
                    || EqPolynomial::evals_cached(w_in),
                );
                Self {
                    current_index: w.len(),
                    current_scalar: F::one(),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                // For high-to-low binding, we bind from MSB (index 0) to LSB (index n-1).
                // The split should be: w_in = first half, w_out = second half
                // [w_first, w_in, w_out]
                let (_, wprime) = w.split_first().unwrap();
                let m = w.len() / 2;
                let (w_in, w_out) = wprime.split_at(m);
                let (E_in_vec, E_out_vec) = rayon::join(
                    || EqPolynomial::evals_cached_rev(w_in),
                    || EqPolynomial::evals_cached_rev(w_out),
                );

                Self {
                    current_index: 0, // Start from 0 for high-to-low up to w.len() - 1
                    current_scalar: F::one(),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
        }
    }

    /// ```ignore (idk why it tries to run doctest)
    ///  Compute the split equality polynomial for the small value optimization
    ///
    /// The split is done as follows: (here `l = num_small_value_rounds`)
    ///
    /// 0 ..... (n/2 - l) ..... (n - l) ..... n
    ///
    ///           <-- E_in -->
    ///
    /// E_out --->                <--- E_out
    ///
    /// where the first E_out part (0 to n/2 - l) corresponds to x_out, and the second E_out part
    /// (n/2 - l to n) corresponds to y_suffix
    ///
    /// Returns E_out which contains `l` vectors of eq evals for the same x_out part, with decreasing
    /// length for y_suffix, and E_in which contains the single vector of eq evals for the x_in part.
    ///
    /// Note the differences between this and the `new` constructor: this is specialized for the
    /// small value optimization.
    pub fn new_for_small_value(
        w: &[F],
        num_x_out_vars: usize,
        num_x_in_vars: usize,
        num_small_value_rounds: usize,
    ) -> Self {
        // Split w into the slices: (l = num_small_value_rounds)
        // (n/2 - l) ..... (n - l)
        // 0..(n/2 - l - 1) concatenated with (n - l)...n
        // Then invoke the evals_cached constructor on the concatenated slice, producing E_out
        // Invoke the evals constructor (no caching) on the middle slice, producing E_in
        // In other words, there is only 1 vector in E_in, and l vectors in E_out
        // (we may drop the rest of the vectors after evals_cached)
        let n = w.len();

        assert!(
            n > 0,
            "length of w must be positive for the split to be valid."
        );
        assert!(num_x_out_vars + num_x_in_vars + num_small_value_rounds == n, "num_x_out_vars + num_x_in_vars + num_small_value_rounds must be == n for the split to be valid.");

        // This should be `min(num_steps, n/2 - num_small_value_rounds)`, computed externally before calling this function.
        let split_point_x_out = num_x_out_vars;
        let split_point_x_in = split_point_x_out + num_x_in_vars;

        let w_E_in_vars: Vec<F> = w[split_point_x_out..split_point_x_in].to_vec();

        // Determine the end index for the suffix part of w_E_out_vars
        let suffix_slice_end = if num_small_value_rounds == 0 {
            split_point_x_in // Results in an empty suffix, e.g., w[n..n]
        } else {
            n - 1 // Use up to n-1, excluding the last variable of w (tau)
        };

        let num_actual_suffix_vars = suffix_slice_end.saturating_sub(split_point_x_in);

        let mut w_E_out_vars: Vec<F> = Vec::with_capacity(num_x_out_vars + num_actual_suffix_vars);
        w_E_out_vars.extend_from_slice(&w[0..split_point_x_out]);
        if split_point_x_in < suffix_slice_end {
            // Add suffix only if range is valid and non-empty
            w_E_out_vars.extend_from_slice(&w[split_point_x_in..suffix_slice_end]);
        }

        // Do not scale E_in; we correct the typed unreduced accumulation with inv(K) after reduction.
        let (mut E_out_vec, E_in) = rayon::join(
            || EqPolynomial::evals_cached(&w_E_out_vars),
            || EqPolynomial::evals(&w_E_in_vars),
        );

        // Take only the first `num_small_value_rounds` vectors from E_out_vec (after reversing)
        // Recall that at this point, E_out_vec[0] = `eq(w[0..split_point_x_out] ++ w[split_point_x_in..n-1], x)`
        E_out_vec.reverse();
        E_out_vec.truncate(num_small_value_rounds);

        Self {
            current_index: num_x_out_vars,
            current_scalar: F::one(),
            w: w.to_vec(),
            E_in_vec: vec![E_in],
            E_out_vec,
            binding_order: BindingOrder::LowToHigh, // Small value optimization is always low-to-high
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => 1 << self.current_index,
            BindingOrder::HighToLow => 1 << (self.w.len() - self.current_index),
        }
    }

    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().map_or(0, |v| v.len())
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().map_or(0, |v| v.len())
    }

    /// Return the last vector from `E1` as a slice
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().map_or(&[], |v| v.as_slice())
    }

    /// Return the last vector from `E2` as a slice
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().map_or(&[], |v| v.as_slice())
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index - 1] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index - 1] - r + prod_w_r + prod_w_r;
                // decrement `current_index`
                self.current_index -= 1;
                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                if self.w.len() / 2 < self.current_index {
                    self.E_in_vec.pop();
                } else if 0 < self.current_index {
                    self.E_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index] - r + prod_w_r + prod_w_r;

                // increment `current_index` (going from 0 to n-1)
                self.current_index += 1;

                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                // For high-to-low, we bind variables in the first half first (E_in),
                // then variables in the second half (E_out)
                if self.current_index <= self.w.len() / 2 {
                    // We're binding variables from the first half (E_in)
                    self.E_in_vec.pop();
                } else if self.current_index <= self.w.len() {
                    // We're binding variables from the second half (E_out)
                    self.E_out_vec.pop();
                }
            }
        }
    }

    /// Compute the cubic sumcheck evaluations (i.e., the evaluations at {0, 2, 3}) of a
    /// polynomial s(X) = l(X) * q(X), where l(X) is the current (linear) eq polynomial and
    /// q(X) = c + dX + eX^2, given the following:
    /// - c, the constant term of q
    /// - e, the quadratic term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_evals_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> [F; 3] {
        // We want to compute the evaluations of the cubic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is quadratic, at the points {0, 2, 3}.
        //
        // At this point, we have
        // - the linear polynomial, l(X) = a + bX
        // - the quadratic polynomial, q(X) = c + dX + eX^2
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d + e)
        //
        // Both l and q are represented by their evaluations at 0 and infinity. I.e., we have a, b, c,
        // and e, but not d. We compute s by first computing l and q at points 2 and 3.

        // Evaluations of the linear polynomial
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.w[self.current_index - 1],
                BindingOrder::HighToLow => self.w[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        // Evaluations of the quadratic polynomial
        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        // q(1) = c + d + e
        let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
        // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        [
            cubic_eval_0,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ]
    }

    /// Compute the quadratic sumcheck evaluations (i.e., the evaluations at {0, 2, 3}) of a
    /// polynomial s(X) = l(X) * q(X), where l(X) is the current (linear) Dao-Thaler eq polynomial and
    /// q(X) = c + dx
    /// - c, the constant term of q
    /// - the previous round claim, s(0) + s(1)
    ///
    /// important: This assumes `HighToLow` ordering (as used in the stage 5 batching sumcheck)
    pub fn gruen_evals_deg_2(&self, q_0: F, previous_claim: F) -> [F; 2] {
        assert_eq!(self.binding_order, BindingOrder::HighToLow);
        // We want to compute the evaluations of the quadratic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is linear, at the points {0, 2}.
        //
        // At this point, we have:
        // - the linear polynomial, l(X) = a + bX
        // - the linear polynomial, q(X) = c + dX
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d)
        //
        // We have q(0) = c, and we need to compute q(1) and q(2).

        // Evaluations of the linear eq polynomial
        // For high-to-low, we bind from index 0 upward, hence we take wi = self.w[self.current_index] here
        let eq_eval_1 = self.current_scalar * self.w[self.current_index];
        let eq_eval_0 = self.current_scalar - eq_eval_1;

        // slope for eq
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        // Evaluations of the linear q(x) polynomial
        let linear_eval_0 = q_0;
        let quadratic_eval_0 = eq_eval_0 * linear_eval_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;

        // get q(1) = c + d:
        let linear_eval_1 = quadratic_eval_1 * eq_eval_1.inverse().unwrap();

        // q(2) = c + 2d = 2*q(1) - q(0)
        let linear_eval_2 = (linear_eval_1 + linear_eval_1) - linear_eval_0;

        [quadratic_eval_0, eq_eval_2 * linear_eval_2]
    }

    pub fn merge(&self) -> DensePolynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => {
                // For low-to-high, current_index tracks how many variables remain unbound
                // We want eq(w[0..current_index], x)
                EqPolynomial::evals_parallel(
                    &self.w[..self.current_index],
                    Some(self.current_scalar),
                )
            }
            BindingOrder::HighToLow => {
                // For high-to-low, current_index tracks how many variables have been bound
                // We want eq(w[current_index..], x)
                EqPolynomial::evals_parallel(
                    &self.w[self.current_index..],
                    Some(self.current_scalar),
                )
            }
        };
        DensePolynomial::new(evals)
    }

    pub fn get_current_scalar(&self) -> F {
        self.current_scalar
    }

    pub fn get_current_w(&self) -> F {
        self.w[self.current_index - 1]
    }
}
