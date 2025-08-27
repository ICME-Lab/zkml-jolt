use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum RightMSB<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for RightMSB<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() < WORD_SIZE {
            return 1;
        }

        (((u64::from(b) % (1 << WORD_SIZE)) >> (WORD_SIZE - 1)) & 1) as u32
    }
}
