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

        // if x < 0, abs(x) = -x = (!x) + 1
        let mut negative_case = F::one();
        for i in 0..WORD_SIZE - 1 {
            negative_case += F::from_u64(1 << i) * (F::one() - r[r.len() - 1 - i]);
        }

        // Keep positive
        positive_case * (F::one() - r[WORD_SIZE]) + negative_case * r[WORD_SIZE]
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for Abs<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Relu, Suffixes::AbsNegativeCase]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, relu, abs_negative_case] = suffixes.try_into().unwrap();

        prefixes[Prefixes::Abs] * one
            + prefixes[Prefixes::NotUnaryMsb] * relu
            + prefixes[Prefixes::UnaryMsb] * abs_negative_case
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_ff::AdditiveGroup;
    use rand::prelude::*;

    use crate::{
        field::JoltField,
        jolt::lookup_table::{
            prefixes::Prefixes,
            test::{
                lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test,
                prefix_suffix_on_hypercube, prefix_suffix_test,
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
    fn test_prefix_suffix_hypercube() {
        let lookup_index = -5i32 as u32 as u64;
        // the operands that are multiplied in `combine`
        let prefix_suffix_combinations = [
            (Prefixes::Abs as usize, 0),
            (Prefixes::NotUnaryMsb as usize, 1),
            (Prefixes::UnaryMsb as usize, 2),
        ];
        prefix_suffix_on_hypercube::<Fr, Abs<32>>(lookup_index, Some(&prefix_suffix_combinations));
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

        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..1000 {
            let x = rng.next_u64();

            let abs_number = abs.materialize_entry(x);
            assert_eq!(
                abs_number,
                (x as u32 as i32).unsigned_abs() as u64,
                "abs({x}) = {abs_number}, expected {}",
                (x as u32 as i32).abs()
            );
        }
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

        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..1000 {
            let x = rng.next_u64();
            let r = int_to_field_bits::<32>(x);
            let mle = abs.evaluate_mle::<Fr>(&r);
            assert_eq!(
                mle,
                Fr::from_u64((x as u32 as i32).unsigned_abs() as u64),
                "abs({x}) = {mle}, expected {}",
                (x as u32 as i32).abs()
            );
        }
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
