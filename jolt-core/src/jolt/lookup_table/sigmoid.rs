use std::f64::consts::E;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::poly::eq_poly::EqPolynomial;
use crate::jolt::lookup_table::prefixes::{PrefixEval, Prefixes};
use crate::jolt::lookup_table::suffixes::{SuffixEval, Suffixes};
use crate::jolt::lookup_table::JoltLookupTable;
use crate::jolt::lookup_table::PrefixSuffixDecomposition;
use crate::field::JoltField;

const LUT_SIZE: usize = 112;
pub const SCALE: f32 = 7.;
pub const SIGMOID_SCALED_TABLE: [u8; LUT_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];

pub const APPROXIMATE_SIGMOID_SCALED_TABLE: [u8; LUT_SIZE] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2,
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for SigmoidTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let max = 1 << WORD_SIZE;
        let i = index % max;
        if i < (LUT_SIZE / 2) as u64 {
            APPROXIMATE_SIGMOID_SCALED_TABLE[i as usize + LUT_SIZE / 2] as u64
        } else if i > max - (LUT_SIZE / 2) as u64 {
            let diff = max - i;
            APPROXIMATE_SIGMOID_SCALED_TABLE[(LUT_SIZE / 2 - diff as usize) - 1] as u64
        } else if i < max / 2 {
            SCALE as u64
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        println!("r = {:?}", r);

        let is_neg = r[0];

        let mut is_small = F::one();
        // We only consider the first 4 bits of the input to sigmoid
        for i in 0..2 * WORD_SIZE - 1 - 3 {
            is_small *= F::one() - r[i];
        }

        let mut is_big = F::one();
        for i in 0..2 * WORD_SIZE - 1 - 3 {
            is_big *= r[i];
        }

        let mut pos_value = F::from_u8(4);
        // The first two bits don't affect the result of sigmoid
        // for i in 0..2 {
        //     index += F::from_u64(1 << i) * r[WORD_SIZE - 1 - i];
        // }

        for i in 2..4 {
            pos_value += r[2 * WORD_SIZE - 1 - i];
        }

        let mut neg_value = F::from_u8(3);
        for i in 2..4 {
            neg_value -= r[2 * WORD_SIZE - 1 - i];
        }

        // println!("neg_value = {:?}", neg_value);
        // println!("pos_value = {:?}", pos_value);
        // println!("is_big = {:?}", is_big);
        // println!("is_small = {:?}", is_small);
        // println!("is_neg = {:?}", is_neg);
        neg_value * is_big * is_neg + pos_value * is_small * (F::one() - is_neg) + F::from_u8(SCALE as u8) * (F::one() - is_small) * (F::one() - is_neg)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for SigmoidTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::SigmoidTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, SigmoidTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, SigmoidTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, SigmoidTable<32>>();
    }
}
