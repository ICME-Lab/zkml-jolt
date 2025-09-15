use crate::field::JoltField;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Abs<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for Abs<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (WORD_SIZE - 1);
        if sign_bit & index == 0 {
            index % (1 << WORD_SIZE)
        } else {
            // In two's complement, -x = (!x) + 1
            // Where ! is the bitwise NOT operator
            ((!index).wrapping_add(1)) % (1 << WORD_SIZE)
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut positive_case = F::zero();
        for i in 0..WORD_SIZE - 1 {
            positive_case += F::from_u64(1 << i) * r[r.len() - 1 - i];
        }

        let mut negative_case = F::one();
        for i in 0..WORD_SIZE - 1 {
            negative_case += F::from_u64(1 << i) * (F::one() - r[r.len() - 1 - i]);
        }

        // Keep positive
        positive_case * (F::one() - r[WORD_SIZE]) + negative_case * r[WORD_SIZE]
    }
}

// TODO(AntoineF4C5): Implement Abs suffix/prefix
impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for Abs<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![
            Suffixes::One,
            Suffixes::AbsNegativeCase,
            Suffixes::LowerWord,
        ]
    }

    // TODO(AntoineF4C5): Does not work yet - Unsure if valid expression
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, abs_negative_case, lower_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Abs] * one
            + prefixes[Prefixes::NotUnaryMsb] * lower_word
            + (F::one() - prefixes[Prefixes::NotUnaryMsb]) * abs_negative_case
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;

    use crate::{
        field::JoltField,
        jolt::lookup_table::{
            test::{
                lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test,
                prefix_suffix_test,
            },
            JoltLookupTable,
        },
    };

    use super::Abs;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, Abs<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, Abs<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, Abs<32>>();
    }

    #[test]
    fn materialize_entry_abs() {
        let abs = Abs::<32>;
        let abs_number = abs.materialize_entry(-1i32 as u32 as u64);
        assert_eq!(abs_number, 1); // abs(-1) = 1

        let abs_number = abs.materialize_entry(1);
        assert_eq!(abs_number, 1); // abs(1) = 1

        let abs_number = abs.materialize_entry(0);
        assert_eq!(abs_number, 0); // abs(0) = 0

        // i32::MIN = -2^31
        let abs_number = abs.materialize_entry((-2i64.pow(31) as i32) as u64);
        assert_eq!(abs_number, 2u64.pow(31)); // abs(-2^31) = 2^31

        // i32::MAX = 2^31 - 1
        let abs_number = abs.materialize_entry((2i64.pow(31) - 1) as i32 as u64);
        assert_eq!(abs_number, 2u64.pow(31) - 1); // abs(2^31 - 1) = 2^31 - 1
    }

    #[test]
    fn evaluate_mle_abs() {
        let abs = Abs::<32>;

        let r = int_to_field_bits::<32>(-1i32 as u32 as u64);
        let mle = abs.evaluate_mle::<Fr>(&r);
        assert_eq!(mle, Fr::from_u64(1)); // abs(-1) = 1

        let r = int_to_field_bits::<32>(1);
        let mle = abs.evaluate_mle::<Fr>(&r);
        assert_eq!(mle, Fr::from_u64(1)); // abs(1) = 1

        let r = int_to_field_bits::<32>(0);
        let mle = abs.evaluate_mle::<Fr>(&r);
        assert_eq!(mle, Fr::from_u64(0)); // abs(0) = 0

        // i32::MIN = -2^31
        let r = int_to_field_bits::<32>(-2i64.pow(31) as i32 as u64);
        let mle = abs.evaluate_mle::<Fr>(&r);
        assert_eq!(mle, Fr::from_u64(2u64.pow(31))); // abs(-2^31) = 2^31

        // i32::MAX = 2^31 - 1
        let r = int_to_field_bits::<32>((2i64.pow(31) - 1) as i32 as u64);
        let mle = abs.evaluate_mle::<Fr>(&r);
        assert_eq!(mle, Fr::from_u64(2u64.pow(31) - 1)); // abs(2^31 - 1) = 2^31 - 1
    }

    fn int_to_field_bits<const WORD_SIZE: usize>(number: u64) -> Vec<Fr> {
        let mut r = vec![Fr::ZERO; 2 * WORD_SIZE];
        let len = 2 * WORD_SIZE;
        for i in 0..len {
            r[len - 1 - i] = Fr::from_u64(((number & (1 << i)) != 0) as u64);
        }
        r
    }
}
