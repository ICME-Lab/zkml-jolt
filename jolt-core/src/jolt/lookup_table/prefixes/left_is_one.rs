use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftOperandIsOnePrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftOperandIsOnePrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F {
        let (_, y) = b.uninterleave();
        // Short-circuit if low-order bits of `y` are not 0s
        if u64::from(y) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::LeftOperandIsOne].unwrap_or(F::one());

        if r_x.is_some() {
            let y = F::from_u8(c as u8);
            result *= y;
        } else {
            let y = F::from_u8(b.pop_msb());
            result *= y;
        }
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _: F,
        r_y: F,
        _: usize,
    ) -> PrefixCheckpoint<F> {
        // checkpoint *= 1
        let updated =
            checkpoints[Prefixes::LeftOperandIsOne].unwrap_or(F::zero()) * r_y;
        Some(updated).into()
    }
}
