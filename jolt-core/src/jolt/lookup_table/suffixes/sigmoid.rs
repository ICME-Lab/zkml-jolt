use crate::{jolt::lookup_table::sigmoid::{APPROXIMATE_SIGMOID_SCALED_TABLE, LUT_SIZE, SCALE}, subprotocols::sparse_dense_shout::LookupBits};

use super::SparseDenseSuffix;

/// Sigmoid suffix
pub enum SigmoidSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for SigmoidSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let max = 1 << WORD_SIZE;
        let i = u64::from(b);
        if i < (LUT_SIZE / 2) as u64 {
            APPROXIMATE_SIGMOID_SCALED_TABLE[i as usize + LUT_SIZE / 2] as u32
        } else if i > max - (LUT_SIZE / 2) as u64 {
            let diff = max - i;
            APPROXIMATE_SIGMOID_SCALED_TABLE[(LUT_SIZE / 2 - diff as usize) - 1] as u32
        } else if i < max / 2 {
            SCALE as u32
        } else {
            0
        }
    }
}
