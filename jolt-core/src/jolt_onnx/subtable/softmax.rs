//! Lookup table for sigmoid of positive values.

use itertools::Itertools;

use crate::{field::JoltField, jolt::subtable::LassoSubtable, poly::eq_poly::EqPolynomial};
use std::marker::PhantomData;

/// A lookup table that returns the softmax of a value.
#[derive(Default)]
pub struct SoftmaxSubtable<F: JoltField> {
    _field: PhantomData<F>,
    max: i8,
}

impl<F: JoltField> SoftmaxSubtable<F> {
    /// Creates a new instance of [`SoftmaxSubtable`].
    pub fn new(max: i8) -> Self {
        Self {
            _field: PhantomData,
            max,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for SoftmaxSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        todo!()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::softmax::SoftmaxSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        softmax_materialize_mle_parity,
        SoftmaxSubtable<Fr>,
        Fr,
        256
    );
}
