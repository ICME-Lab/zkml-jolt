//! Lookup table for sigmoid of positive values.

use itertools::Itertools;
use num_traits::Pow;

use crate::{field::JoltField, jolt::subtable::LassoSubtable, poly::eq_poly::EqPolynomial};
use std::marker::PhantomData;


/// A lookup table that returns the sigmoid of a value.
#[derive(Default)]
pub struct Pow2Subtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> Pow2Subtable<F> {
    /// Creates a new instance of [`Pow2Subtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for Pow2Subtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        for i in 0..M {
            entries[i] = 2u32.pow(i as u32) as u32;
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        let mut f_eval: Vec<F> = vec![F::from_u8(255); 1 << point.len()];
        todo!();

        let eq_evals = EqPolynomial::evals(point);
        f_eval
            .iter()
            .zip_eq(eq_evals.iter())
            .map(|(x, e)| *x * e)
            .sum()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::pow_2::Pow2Subtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        pow_2_materialize_mle_parity,
        Pow2Subtable<Fr>,
        Fr,
        256
    );
}
