use crate::{jolt::lookup_table::sigmoid::{APPROXIMATE_SIGMOID_SCALED_TABLE, LUT_SIZE, SCALE}, subprotocols::sparse_dense_shout::LookupBits};

use super::SparseDenseSuffix;

/// Sigmoid suffix
pub enum PositiveSigmoidSuffix<const WORD_SIZE: usize> {}

// TODO: Note that the bit chunks are smaller than in the lookup table case
// TODO: Maybe we need a negative sigmoid and a positive sigmoid
impl<const WORD_SIZE: usize> SparseDenseSuffix for PositiveSigmoidSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() > 4 {
            return 0;
        } else {
            APPROXIMATE_SIGMOID_SCALED_TABLE[usize::from(b) + LUT_SIZE / 2] as u32
        }
    }
}
