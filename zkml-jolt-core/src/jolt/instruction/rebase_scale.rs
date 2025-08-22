// TODO

// use common::constants::virtual_register_index;
// use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

// use super::VirtualInstructionSequence;
// use crate::jolt::instruction::{
//     add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
//     virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
//     virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
// };

use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::{
    jolt::instruction::{VirtualInstructionSequence, div::DIVInstruction},
    utils::u64_vec_to_i32_iter,
};

macro_rules! expect_rebase_scale {
    ($cycle:expr) => {
        match $cycle.instr.opcode {
            ONNXOpcode::RebaseScale(_) => {}
            _ => panic!("Expected ONNXOpcode::RebaseScale"),
        }
    };
}

/// Perform signed division and return the result
pub struct REBASEInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for REBASEInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 2;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        expect_rebase_scale!(cycle);

        let inner_opcode = match &cycle.instr.opcode {
            ONNXOpcode::RebaseScale(inner) => inner,
            _ => unreachable!(),
        };

        // Virtual registers used in sequence
        // TODO(AntoineF4C5): Check if conflict between this and inner DIV's virtual register also indexed to 0
        let v_0 = Some(virtual_tensor_index(0)); // Inner operator output to be rebased 

        let mut virtual_trace = vec![];

        // Apply inner operator
        let inner_opcode = (**inner_opcode).clone();
        let mut instr = cycle.instr.clone();
        instr.opcode = inner_opcode;
        let inner_res_0: Vec<u64> = Vec::new(); // TODO(AntoineF4C5): Need to set to inner operator's result
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address, // TODO(AntoineF4C5): Might be more readable to have separate behavior for each possible inner opcode
                opcode: instr.opcode,         // and only set required fields
                ts1: cycle.instr.ts1,
                ts2: cycle.instr.ts2,
                ts3: cycle.instr.ts3,
                td: v_0,
                imm: cycle.instr.imm.clone(),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: cycle.memory_state.ts2_val.clone(),
                ts3_val: cycle.memory_state.ts3_val.clone(),
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))), // Needs to set to the inner operator's expected output
            },
            advice_value: None,
        });

        // Apply div operator by 2^scale
        let res = DIVInstruction::<WORD_SIZE>::sequence_output(
            inner_res_0.clone(),
            vec![128; MAX_TENSOR_SIZE], // TODO(AntoineF4C5): Check if 0 for anything out of active_output_elements
        );
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Div,
                ts1: v_0,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                // TODO(AntoineF4C5): Check if 0 for anything out of active_output_elements
                // Do we need to enforce 128 denominator
                imm: Some(Tensor::from(u64_vec_to_i32_iter(&vec![
                    128;
                    MAX_TENSOR_SIZE
                ]))),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: cycle.instr.active_output_elements,
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(u64_vec_to_i32_iter(&inner_res_0))),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(Tensor::from(u64_vec_to_i32_iter(&res))),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(_x: Vec<u64>, _y: Vec<u64>) -> Vec<u64> {
        todo!()
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

//     #[test]
//     fn div_virtual_sequence_32() {
//         jolt_virtual_sequence_test!(DIVInstruction::<32>, RV32IM::DIV);
//     }
// }
