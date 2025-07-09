//! This module defines the [`ONNXJoltVM`] type and its instruction set, which is a Jolt VM & ISA for ONNX models.

use super::JoltProof;
use crate::field::JoltField;
use crate::jolt::instruction::add::ADDInstruction;
use crate::jolt::instruction::beq::BEQInstruction;
use crate::jolt::instruction::mul::MULInstruction;
use crate::jolt::instruction::virtual_advice::ADVICEInstruction;
use crate::jolt::instruction::virtual_assert_valid_div0::AssertValidDiv0Instruction;
use crate::jolt::instruction::virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction;
use crate::jolt::instruction::virtual_move::MOVEInstruction;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt_onnx::common::onnx_trace::{ONNXInstruction, ONNXTraceRow, Operator};
use crate::jolt_onnx::instruction::max::MaxInstruction;
use crate::jolt_onnx::instruction::pow_2::Pow2Instruction;
use crate::jolt_onnx::instruction::JoltONNXInstructionSet;
use crate::jolt::subtable::{
    identity::IdentitySubtable, JoltSubtableSet, LassoSubtable, SubtableId,
};
use crate::jolt_onnx::{instruction::{relu::ReLUInstruction, sigmoid::SigmoidInstruction}, subtable::is_pos::IsPosSubtable};
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types, missing_docs)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, PartialEq, EnumIter, EnumCountMacro, Serialize, Deserialize)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name {
            $($alias($struct)),+
        }
        impl JoltONNXInstructionSet for $enum_name {}
        impl $enum_name {
            /// Create a random instruction from the enum.
            pub fn random_instruction(rng: &mut StdRng) -> Self {
                let index = rng.next_u64() as usize % $enum_name::COUNT;
                let instruction = $enum_name::iter()
                    .enumerate()
                    .filter(|(i, _)| *i == index)
                    .map(|(_, x)| x)
                    .next()
                    .unwrap();
                instruction.random(rng)
            }
        }
        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        impl Default for $enum_name {
            fn default() -> Self {
                $enum_name::iter().collect::<Vec<_>>()[0]
            }
        }
    };
}

/// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
/// are callable on the enum type via enum_dispatch.
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types, missing_docs)]
        #[repr(u8)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: JoltField> { $($alias($struct)),+ }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) }
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
                let byte = unsafe { *(&subtable as *const $enum_name<F> as *const u8) };
                byte as usize
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

/// C constant in Jolt paper
pub const C_ONNX: usize = 4;
/// Size of subtable entries
pub const M_ONNX: usize = 1 << 16;
const WORD_SIZE: usize = 32;

instruction_set!(
  ONNXInstructionSet,
  ReLU: ReLUInstruction,
  ADD: ADDInstruction<WORD_SIZE>,
  MUL: MULInstruction<WORD_SIZE>,
  Sigmoid: SigmoidInstruction,
  VirtualAdvice: ADVICEInstruction<WORD_SIZE>,
  VirtualAssertValidDiv0: AssertValidDiv0Instruction<WORD_SIZE>,
  VirtualAssertValidSignedRemainder: AssertValidSignedRemainderInstruction<WORD_SIZE>,
  VirtualAssertEq: BEQInstruction<WORD_SIZE>,
  VirtualMove: MOVEInstruction<WORD_SIZE>,
  Max: MaxInstruction<WORD_SIZE>,
  Pow2: Pow2Instruction
);

subtable_enum!(
  ONNXSubtables,
  IDENTITY: IdentitySubtable<F>,
  IS_POS: IsPosSubtable<F>
);

/// The ONNX Jolt VM type, which is a Jolt VM for ONNX models.
pub type ONNXJoltVM<F, PCS, ProofTranscript> =
    JoltProof<C_ONNX, M_ONNX, F, PCS, ONNXInstructionSet, ONNXSubtables<F>, ProofTranscript>;



impl TryFrom<&ONNXInstruction> for ONNXInstructionSet {
    type Error = &'static str;
    
    #[rustfmt::skip] 
    fn try_from(instruction: &ONNXInstruction) -> Result<Self, Self::Error> {
        match instruction.opcode {
            _ => Err("No corresponding ONNX instruction")
        }
    }
}

impl TryFrom<&ONNXTraceRow> for ONNXInstructionSet {
    type Error = &'static str;

    #[rustfmt::skip] 
    fn try_from(row: &ONNXTraceRow) -> Result<Self, Self::Error> {
        match row.instruction.opcode {
            Operator::Relu => Ok(ReLUInstruction(row.layer_state.input_vals[0].data[0] as u64).into()),
            Operator::Add => Ok(ADDInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::Mul => Ok(MULInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::Sigmoid => Ok(SigmoidInstruction(row.layer_state.input_vals[0].data[0] as u64).into()),
            Operator::VirtualAdvice => Ok(ADVICEInstruction::<WORD_SIZE>(row.advice_value[0].data[0] as u64).into()),
            Operator::VirtualAssertValidDiv0 => Ok(AssertValidDiv0Instruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::VirtualAssertValidSignedRemainder => Ok(AssertValidSignedRemainderInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::VirtualAssertEq => Ok(BEQInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::VirtualMove => Ok(MOVEInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64).into()),
            Operator::Max => Ok(MaxInstruction::<WORD_SIZE>(row.layer_state.input_vals[0].data[0] as u64, row.layer_state.input_vals[1].data[0] as u64).into()),
            Operator::Pow2 => Ok(Pow2Instruction(row.layer_state.input_vals[0].data[0] as u64).into()),
            _ => Err("No corresponding ONNX instruction")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ONNXJoltVM;
    use crate::jolt_onnx::onnx_host::ONNXProgram;
    use crate::jolt_onnx::utils::random_floatvec;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use crate::{field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme};
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;

    fn test_e2e_with<F, PCS, ProofTranscript>(onnx_program: &ONNXProgram)
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        // Setup model and get trace (input for proving)
        let model = onnx_program.decode();

        // Generate preprocessing
        let pp = ONNXJoltVM::<F, PCS, ProofTranscript>::prover_preprocess(&model, 1 << 20);

        // Prove
        let (io, trace) = onnx_program.trace();
        let (snark, commitments, verifier_io, _) =
            ONNXJoltVM::<F, PCS, ProofTranscript>::prove(io, trace, pp.clone());

        // Verify
        snark
            .verify(pp.shared, commitments, verifier_io, None)
            .unwrap();
    }

    #[test]
    fn test_perceptron() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/perceptron.onnx",
            Some(random_floatvec(&mut test_rng(), 10)),
        ))
    }

    #[test]
    fn test_perceptron_2() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/perceptron_2.onnx",
            Some(random_floatvec(&mut test_rng(), 4)),
        ))
    }

    #[test]
    fn test_accuracy() {
        test_e2e_with::<Fr, HyperKZG<Bn254, KeccakTranscript>, KeccakTranscript>(&ONNXProgram::new(
            "onnx/mlp/accuracy.onnx",
            Some(random_floatvec(&mut test_rng(), 41)),
        ))
    }
}
