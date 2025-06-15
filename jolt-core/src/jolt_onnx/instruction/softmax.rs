//! A sum-check precompile for softmax.
//! The softmax function takes as input a tuple z of K real numbers, 
//! and normalizes it into a probability distribution consisting 
//! of K probabilities proportional to the exponentials of the input numbers.
//! 
//! After applying softmax, each component will be in the interval (0,1), 
//! and the components will add up to 1, so that they can be interpreted 
//! as probabilities.
//! 
//! Formally, the standard (unit) softmax function 
//! s : R^K -> (0,1)^K, where K>1, 
//! takes a tuple z =(z_1, ..., z_K) in R^K 
//! and computes each component of vector s(z) in (0,1)^K with
//! 
//! s(z)_i = e^z_i / sum(e^z_j) for j = 1 to K

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::LassoSubtable;
use crate::jolt_onnx::subtable::is_pos::IsPosSubtable;
use crate::jolt_onnx::subtable::is_zero::IsZeroSubtable;

use crate::jolt_onnx::subtable::softmax::SoftmaxSubtable;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};
use ark_std::log2;
use itertools::Itertools;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

/// Input scale for softmax. Input values are between -8 and 8. Quantized input values are between 0 and 255.
pub const INPUT_SCALE: f32 = 16.0 / 256.0;
/// Quantized input zero
pub const INPUT_ZERO_POINT: i64 = 128;
/// Output scale for softmax. Output values are between 0 and 1. Quantized output values are between 0 and 255.
pub const OUTPUT_SCALE: f32 = 1.0 / 256.0;
/// Quantized output zero
pub const OUTPUT_ZERO_POINT: i64 = 0;

fn vec_i8_to_u64_le(vec: &[i8]) -> u64 {
    let mut bytes = [0u8; 8]; // u64 is 8 bytes
    for (i, &val) in vec.iter().take(8).enumerate() {
        bytes[i] = val as u8;
    }
    u64::from_le_bytes(bytes)
}

fn u64_le_to_vec_i8(val: &u64) -> Vec<i8> {
    let bytes = val.to_le_bytes();
    bytes.iter().map(|&b| b as i8).collect()
}

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SoftmaxInstruction(Vec<i8>);

impl SoftmaxInstruction {
    pub fn new(z: Vec<i8>) -> Self {
        Self(z)
    }

    pub fn max(&self) -> i8 {
        *self.0.iter().max().unwrap()
    }

}

impl JoltInstruction for SoftmaxInstruction {
    fn operands(&self) -> (u64, u64) {
        let packed_input = vec_i8_to_u64_le(&self.0);
        (packed_input, 0)
    }
    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize)
    }
    fn materialize_entry(&self, _index: u64) -> u64 {
        todo!()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        0
    }
    fn subtables<F: JoltField>(&self, C: usize, _M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (
                Box::new(SoftmaxSubtable::<F>::new(self.max())),
                SubtableIndices::from(0..C),
            ),
        ]
    }
    fn random(&self, rng: &mut StdRng) -> Self {
        use rand::Rng;
        let z = (0..self.0.len()).map(|_| rng.gen_range(0,255)).collect();
        Self(z)
    }

    fn to_indices(&self, C: usize, _M: usize) -> Vec<usize> {
        todo!()
    }
    
    fn lookup_entry(&self) -> u64 {
        let n = self.0.len();
        let max = self.max();

        let mut output = vec![0; n];

        let mut normalized_sum = 0;
        for i in 0..n {
            // We are applying safe softmax, so we shift the input by the max value.
            let z_i = (self.0[i] - max) as f32;
            let e_z_i = z_i.exp().round() as u32;
            let e_z_i_i8 = e_z_i.clamp(0, 255) as i8;
            output[i] = e_z_i_i8;
            normalized_sum += e_z_i_i8;
        }

        for i in 0..n {
            output[i] = output[i] / normalized_sum;
        }

        vec_i8_to_u64_le(&output)
    }

    
    fn evaluate_mle<F>(&self, _point: &[F]) -> F
    where
        F: JoltField,
    {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use ark_std::{rand::Rng, test_rng};

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use super::*;

    #[test]
    fn test_softmax_lookup() {
        let softmax = SoftmaxInstruction::new(vec![1, 2, 3]);
        let lookup_entry = softmax.lookup_entry();
        assert_eq!(lookup_entry, 0x01020300);
    }

    #[test]
    fn softmax_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 8;

        // random vector of i8
        let v = (0..8).map(|_| rng.gen_range(0..255) as i8).collect();
        let instruction = SoftmaxInstruction(v);
        jolt_instruction_test!(instruction);
    }
}