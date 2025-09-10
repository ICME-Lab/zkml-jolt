use crate::{jolt::lookup_table::sigmoid::{APPROXIMATE_SIGMOID_SCALED_TABLE, LUT_SIZE, SCALE}, subprotocols::sparse_dense_shout::LookupBits};

use super::SparseDenseSuffix;

/// Sigmoid suffix
pub enum NegativeSigmoidSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for NegativeSigmoidSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() > 4 {
            return SCALE as u32;
        } else {
            APPROXIMATE_SIGMOID_SCALED_TABLE[LUT_SIZE / 2 - usize::from(b) - 1] as u32
        }
    }
}
