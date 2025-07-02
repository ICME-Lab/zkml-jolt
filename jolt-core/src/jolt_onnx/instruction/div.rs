use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use crate::{jolt::instruction::{
    add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
    virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
}, jolt_onnx::{common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator}, instruction::VirtualInstructionSequence}};
/// Perform signed division and return the result
pub struct DIVInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for DIVInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 8;

    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow> {
        assert_eq!(trace_row.instruction.opcode, Operator::Div);
        // DIV source tensor references
        let r_x = trace_row.instruction.input_refs[0];
        let r_y = trace_row.instruction.input_refs[1];
        // Virtual references used in sequence
        // TODO: Not sure if this is the best way to do this.
        let v_0 = "v_0".to_string();
        let v_q = "v_q".to_string();
        let v_r = "v_r".to_string();
        let v_qy = "v_qy".to_string();
        // DIV operands
        // TODO: Do we want to have entry-wise division?
        let x = trace_row.layer_state.input_vals.as_ref()[0].data[0];
        let y = trace_row.layer_state.input_vals.as_ref()[1].data[1];

        let mut virtual_trace = vec![];

        let (quotient, remainder) = match WORD_SIZE {
            32 => {
                if y == 0 {
                    (u32::MAX as u64, x)
                } else {
                    let mut quotient = x as i32 / y as i32;
                    let mut remainder = x as i32 % y as i32;
                    if (remainder < 0 && (y as i32) > 0) || (remainder > 0 && (y as i32) < 0) {
                        remainder += y as i32;
                        quotient -= 1;
                    }
                    (quotient as u32 as u64, remainder as u32 as u64)
                }
            }
            64 => {
                if y == 0 {
                    (u64::MAX, x)
                } else {
                    let mut quotient = x as i64 / y as i64;
                    let mut remainder = x as i64 % y as i64;
                    if (remainder < 0 && (y as i64) > 0) || (remainder > 0 && (y as i64) < 0) {
                        remainder += y as i64;
                        quotient -= 1;
                    }
                    (quotient as u64, remainder as u64)
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {WORD_SIZE}"),
        };

        let q = ADVICEInstruction::<WORD_SIZE>(quotient).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAdvice,
                attributes: None,
                input_refs: vec![],
                output_refs: vec![v_q],
            },
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![q],
            },
            // advice_value: Some(quotient),
        });

        let r = ADVICEInstruction::<WORD_SIZE>(remainder).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAdvice,
                attributes: None,
                input_refs: vec![],
                output_refs: vec![v_r],
            },
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![r],
            },
            // advice_value: Some(remainder),
        });

        let is_valid: u64 = AssertValidSignedRemainderInstruction::<WORD_SIZE>(r, y).lookup_entry();
        assert_eq!(is_valid, 1);
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertValidSignedRemainder,
                attributes: None,
                input_refs: vec![v_r, r_y],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![r, y],
                output_vals: vec![],
            },
        });

        let is_valid: u64 = AssertValidDiv0Instruction::<WORD_SIZE>(y, q).lookup_entry();
        assert_eq!(is_valid, 1);
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertValidDiv0,
                attributes: None,
                input_refs: vec![r_y, v_q],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![y, q],
                output_vals: vec![],
            },
        });

        let q_y = MULInstruction::<WORD_SIZE>(q, y).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::Mul,
                attributes: None,
                input_refs: vec![v_q, r_y],
                output_refs: vec![v_qy],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![q, y],
                output_vals: vec![q_y],
            },
        });

        let add_0 = ADDInstruction::<WORD_SIZE>(q_y, r).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::Add,
                attributes: None,
                input_refs: vec![v_qy, v_r],
                output_refs: vec![v_0],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![q_y, r],
                output_vals: vec![add_0],
            },
        });

        let _assert_eq = BEQInstruction::<WORD_SIZE>(add_0, x).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertEq,
                attributes: None,
                input_refs: vec![v_0, r_x],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![add_0, x],
                output_vals: vec![],
            },
        });

        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualMove,
                attributes: None,
                input_refs: vec![v_q],
                output_refs: vec![trace_row.instruction.output_refs[0]],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![q],
                output_vals: vec![q],
            },
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        let x = x as i32;
        let y = y as i32;
        if y == 0 {
            return (1 << WORD_SIZE) - 1;
        }
        let mut quotient = x / y;
        let remainder = x % y;
        if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
            quotient -= 1;
        }
        quotient as u32 as u64
    }
}

#[cfg(test)]
mod test {

    use crate::jolt_onnx::instruction::test::jolt_onnx_virtual_sequence_test;

    use super::*;

    #[test]
    fn div_virtual_sequence_32() {
        jolt_onnx_virtual_sequence_test::<DIVInstruction<32>>(Operator::Div);
    }
}
