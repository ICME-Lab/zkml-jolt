use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum AbsPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for AbsPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        // Ignore high-order variables
        if j < WORD_SIZE {
            return F::zero();
        }
        let nsign_bit =
            *Prefixes::NotUnaryMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        let word = *Prefixes::LowerWordNoMsb.prefix_mle::<WORD_SIZE, F>(checkpoints, r_x, c, b, j);
        let word_two_complement = F::from_u64(1 << (WORD_SIZE - 1)) - word; // Word without sign bit is only WORD_SIZE - 1 bits, so 2's complement is 2^(WORD_SIZE-1) - word

        // nsign_bit * word + (1 - nsign_bit) * word_two_complement
        nsign_bit * (word - word_two_complement) + word_two_complement
    }

    // TODO(AntoineF4C5): Verify Abs git prefix
    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let two = 2 * WORD_SIZE;
        match j {
            // suffix handles abs
            j if j < WORD_SIZE => None.into(),
            j if j == WORD_SIZE + 1 => {
                // Sign bit is in r_x
                let sign_bit = r_x;
                let y_shift = two - j - 1;
                let updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << y_shift)
                        * (r_y * (F::one() - sign_bit) + (F::one() - r_y) * sign_bit); // if positive then r_y, else !r_y
                                                                                       // + sign_bit; // TODO(AntoineF4C5): Maybe need to set only at last iteration.
                                                                                       // if negative then +1 for two's complement (two complement of x = !x + 1)
                Some(updated).into()
            }
            // last iteration
            j if j == two - 1 => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                let nsign_bit = checkpoints[Prefixes::NotUnaryMsb].unwrap();
                let updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift)
                        * (r_x * nsign_bit + (F::one() - r_x) * (F::one() - nsign_bit)) // if positive then r_x, else !r_x
                    + F::from_u64(1 << y_shift)
                        * (r_y * nsign_bit + (F::one() - r_y) * (F::one() - nsign_bit)) // if positive then r_y, else !r_y
                    + (F::one() - nsign_bit); // if negative then +1 for two's complement (two complement of x = !x + 1)
                Some(updated).into()
            }
            _ => {
                let x_shift = two - j;
                let y_shift = x_shift - 1;
                let nsign_bit = checkpoints[Prefixes::NotUnaryMsb].unwrap();
                let updated = checkpoints[Prefixes::Abs].unwrap_or(F::zero())
                    + F::from_u64(1 << x_shift)
                        * (r_x * nsign_bit + (F::one() - r_x) * (F::one() - nsign_bit)) // if positive then r_x, else !r_x
                    + F::from_u64(1 << y_shift)
                        * (r_y * nsign_bit + (F::one() - r_y) * (F::one() - nsign_bit)); // if positive then r_y, else !r_y
                Some(updated).into()
            }
        }
    }
}
