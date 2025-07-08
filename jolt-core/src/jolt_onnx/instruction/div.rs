//! Virtual instruction sequence for the DIV instruction.
use crate::{
    jolt::instruction::{
        add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
        virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
        virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
        JoltInstruction,
    },
    jolt_onnx::{
        common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator},
        instruction::VirtualInstructionSequence,
        tracer::tensor::QuantizedTensor,
    },
};

/// Perform signed division and return the result
pub struct DIVInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for DIVInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 8;

    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow> {
        assert_eq!(trace_row.instruction.opcode, Operator::Div);
        // DIV source tensor references
        let r_x = trace_row.instruction.input_refs[0].clone();
        let r_y = trace_row.instruction.input_refs[1].clone();
        // Virtual references used in sequence
        // TODO: Not sure if this is the best way to do this.
        let v_0 = "v_0".to_string();
        let v_q = "v_q".to_string();
        let v_r = "v_r".to_string();
        let v_qy = "v_qy".to_string();
        // DIV operands
        let x = trace_row.layer_state.input_vals[0].data[0];
        let y = trace_row.layer_state.input_vals[1].data[0];

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
                    (quotient as u32 as u64, remainder as i8)
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
                    (quotient as u64, remainder as i8)
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
                output_refs: vec![v_q.clone()],
            },
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![QuantizedTensor::from(q as i8)],
            },
            advice_value: vec![QuantizedTensor::from(quotient as i8)],
        });

        let r = ADVICEInstruction::<WORD_SIZE>(remainder as u64).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAdvice,
                attributes: None,
                input_refs: vec![],
                output_refs: vec![v_r.clone()],
            },
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![QuantizedTensor::from(r as i8)],
            },
            advice_value: vec![QuantizedTensor::from(remainder as i8)],
        });

        let is_valid: u64 =
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(r as u64, y as u64).lookup_entry();
        assert_eq!(is_valid, 1);
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertValidSignedRemainder,
                attributes: None,
                input_refs: vec![v_r.clone(), r_y.clone()],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![
                    QuantizedTensor::from(r as i8),
                    QuantizedTensor::from(y as i8),
                ],
                output_vals: vec![],
            },
            advice_value: vec![],
        });

        let is_valid: u64 = AssertValidDiv0Instruction::<WORD_SIZE>(y as u64, q).lookup_entry();
        assert_eq!(is_valid, 1);
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertValidDiv0,
                attributes: None,
                input_refs: vec![r_y.clone(), v_q.clone()],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![
                    QuantizedTensor::from(y as i8),
                    QuantizedTensor::from(q as i8),
                ],
                output_vals: vec![],
            },
            advice_value: vec![],
        });

        let q_y = MULInstruction::<WORD_SIZE>(q, y as u64).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::Mul,
                attributes: None,
                input_refs: vec![v_q.clone(), r_y.clone()],
                output_refs: vec![v_qy.clone()],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![
                    QuantizedTensor::from(q as i8),
                    QuantizedTensor::from(y as i8),
                ],
                output_vals: vec![QuantizedTensor::from(q_y as i8)],
            },
            advice_value: vec![],
        });

        let add_0 = ADDInstruction::<WORD_SIZE>(q_y, r).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::Add,
                attributes: None,
                input_refs: vec![v_qy.clone(), v_r.clone()],
                output_refs: vec![v_0.clone()],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![
                    QuantizedTensor::from(q_y as i8),
                    QuantizedTensor::from(r as i8),
                ],
                output_vals: vec![QuantizedTensor::from(add_0 as i8)],
            },
            advice_value: vec![],
        });

        let _assert_eq = BEQInstruction::<WORD_SIZE>(add_0, x as u64).lookup_entry();
        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualAssertEq,
                attributes: None,
                input_refs: vec![v_0.clone(), r_x.clone()],
                output_refs: vec![],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![
                    QuantizedTensor::from(add_0 as i8),
                    QuantizedTensor::from(x as i8),
                ],
                output_vals: vec![],
            },
            advice_value: vec![],
        });

        virtual_trace.push(ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode: Operator::VirtualMove,
                attributes: None,
                input_refs: vec![v_q.clone()],
                output_refs: vec![trace_row.instruction.output_refs[0].clone()],
                // virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            layer_state: LayerState {
                input_vals: vec![QuantizedTensor::from(q as i8)],
                output_vals: vec![QuantizedTensor::from(q as i8)],
            },
            advice_value: vec![],
        });

        virtual_trace
    }

    fn sequence_output(x: i8, y: i8) -> i8 {
        let x = x as i32;
        let y = y as i32;
        if y == 0 {
            return -1;
        }
        let mut quotient = x / y;
        let remainder = x % y;
        if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
            quotient -= 1;
        }
        quotient as i8
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
