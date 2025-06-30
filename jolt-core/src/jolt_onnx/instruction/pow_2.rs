//! Sigmoid instruction for Jolt ONNX

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::LassoSubtable;
use crate::jolt_onnx::subtable::pow_2::Pow2Subtable;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::instruction_utils::chunk_operand_usize;
use ark_std::log2;
use ark_ff::Field;
use itertools::Itertools;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

fn naive_pow<F: JoltField>(x: F, pow: u32) -> F {
    if x == F::one() {
        return F::one();
    }
    let mut result = x;
    let y = 2u32.pow(pow);
    for _ in 1..y {
        result = result * x;
    }
    result
}

/// Sigmoid instruction
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2Instruction(pub u64);

impl JoltInstruction for Pow2Instruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let log_m = log2(M);
        vals[0..C].iter().rev().enumerate().fold(F::one(), |acc, (i, x)| {
            let pow_x = naive_pow(*x, log_m * i as u32);
            acc * pow_x
        })
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (
                Box::new(Pow2Subtable::<F>::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn lookup_entry(&self) -> u64 {
        if self.0 >= 64 {
            u64::MAX
        } else {
            2u64.pow(self.0 as u32)
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand::RngCore;
        Self(rng.next_u32() as u64)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        2u64.pow(index as u32)
    }

    fn evaluate_mle<F>(&self, point: &[F]) -> F
    where
        F: JoltField,
    {
        let mut f_eval: Vec<F> = vec![F::from_u8(255); 1 << point.len()];
        todo!();

        let eq_evals = EqPolynomial::evals(point);
        f_eval
            .iter()
            .zip_eq(eq_evals.iter())
            .map(|(x, e)| *x * e)
            .sum()
    }
}

#[cfg(test)]
mod test {
    use super::Pow2Instruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, materialize_entry_test,
    };
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use ark_bn254::Fr;
    use ark_std::rand::RngCore;
    use ark_std::test_rng;

    #[test]
    fn pow_2_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, Pow2Instruction>();
    }

    #[test]
    fn pow_2_materialize_entry() {
        materialize_entry_test::<Fr, Pow2Instruction>();
    }

    #[test]
    fn pow_2_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 6;
        const M: usize = 1 << 5;

        for i in 0..64 {
            println!("instruction: {:?}", i);
            let instruction = Pow2Instruction(i as u64);
            jolt_instruction_test!(instruction);
        }

        // for _ in 0..256 {
        //     let x = rng.next_u32();
        //     let instruction = Pow2Instruction(x as u64);
        //     jolt_instruction_test!(instruction);
        // }

        // let u32_max: u64 = u32::MAX as u64;
        // let instructions = vec![
        //     Pow2Instruction(0),
        //     Pow2Instruction(1),
        //     Pow2Instruction(8374),
        //     Pow2Instruction((-100_i32) as u64),
        //     Pow2Instruction((-1_i32) as u64),
        //     Pow2Instruction(u32_max),
        //     Pow2Instruction(u32_max + 100),
        //     Pow2Instruction(u32_max + (1 << 8)),
        //     Pow2Instruction(1 << 8),
        //     Pow2Instruction(1 << 30),
        // ];
        // for instruction in instructions {
        //     jolt_instruction_test!(instruction);
        // }
    }
}
