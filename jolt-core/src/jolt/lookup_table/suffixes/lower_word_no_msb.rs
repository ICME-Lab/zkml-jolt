use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Returns the lower WORD_SIZE - 1 bits. Used to range-check values to be in
/// the range [0, 2^WORD_SIZE).
pub enum LowerWordNoMsbSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for LowerWordNoMsbSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        (u64::from(b) % (1 << (WORD_SIZE - 1))) as u32
    }
}
