use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use crate::jolt::instruction::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
    virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
};
use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator};
use crate::jolt_onnx::instruction::max::MaxInstruction;
use crate::jolt_onnx::instruction::pow_2::Pow2Instruction;


fn virtual_trace<const WORD_SIZE: usize>(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow> {
    assert_eq!(trace_row.instruction.opcode, Operator::Softmax);

    let mut virtual_trace = vec![];

    // TODO: Is it safe to assume that there is only one input value?
    let input = trace_row.layer_state.input_vals.unwrap()[0];

    let max_val = input.data.iter().fold(0, |acc, a| {
       let max = MaxInstruction::<WORD_SIZE>(acc, *a as u64).lookup_entry();
       virtual_trace.push(
          ONNXTraceRow {
            instruction: ONNXInstruction::new(Operator::Max),
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![]
            }
          }
       )
    });

    let normalised: Vec<u64> = input.data.iter().map(|z| {
        let a = MULInstruction::<WORD_SIZE>(*z as u64, 63 as u64).lookup_entry();
        virtual_trace.push(
          ONNXTraceRow {
            instruction: ONNXInstruction::new(Operator::Mul),
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![]
            }
          }
       );
       let b = DIVInstruction::<WORD_SIZE>(a, max_val).lookup_entry();
        virtual_trace.push(
          ONNXTraceRow {
            instruction: ONNXInstruction::new(Operator::Div),
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![]
            }
          }
       ); 

       let pow_2 = Pow2Instruction(b).lookup_entry();
        virtual_trace.push(
          ONNXTraceRow {
            instruction: ONNXInstruction::new(Operator::Pow2),
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![]
            }
          }
        );
    }).collect();

// 4. Run the sum-check to prove that $\sum a_i = N$, where $N$ is the normalisation factor.
// 5. Run a division lookup $a_i / N$ for each element in $\vec{a}$. Since the value is in (0,1), we multiply by $2^8$ for quantization.
// 6. Concatenate the results.

    todo!()
}


#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn div_virtual_sequence_32() {
        jolt_virtual_sequence_test::<DIVInstruction<32>>(RV32IM::DIV);
    }
}
