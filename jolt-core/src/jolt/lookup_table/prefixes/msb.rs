use crate::{
    field::JoltField, jolt::lookup_table::prefixes::Prefixes,
    subprotocols::sparse_dense_shout::LookupBits,
};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum MsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for MsbPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        _: LookupBits,
        j: usize,
    ) -> F {
        match j {
            j if j == WORD_SIZE => F::from_u32(c),
            j if j == WORD_SIZE + 1 => r_x.unwrap(),
            j if j < WORD_SIZE => F::one(),
            _ => checkpoints[Prefixes::Msb].unwrap(),
        }
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        match j {
            j if j == WORD_SIZE => Some(r_y).into(),
            j if j == WORD_SIZE + 1 => Some(r_x).into(),
            j if j < WORD_SIZE => Some(F::one()).into(),
            _ => checkpoints[Prefixes::Msb].into(),
        }
    }
}
