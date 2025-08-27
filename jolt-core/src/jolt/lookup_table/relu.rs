use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReLUTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for ReLUTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (WORD_SIZE - 1);
        if index & sign_bit == 0 {
            index % (1 << WORD_SIZE)
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE - 1 {
            result += F::from_u64(1 << i) * r[r.len() - 1 - i];
        }
        result *= F::one() - r[WORD_SIZE];
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for ReLUTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWordNoMsb, Suffixes::RightMSB]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word, rmsb] = suffixes.try_into().unwrap();
        (one - prefixes[Prefixes::Msb] * one * rmsb)
            * (prefixes[Prefixes::LowerWordNoMsb] * one + lower_word)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::ReLUTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, ReLUTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, ReLUTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, ReLUTable<32>>();
    }
}
