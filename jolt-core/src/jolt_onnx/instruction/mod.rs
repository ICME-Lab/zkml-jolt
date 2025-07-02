//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::jolt_onnx::common::onnx_trace::{ONNXInstruction, ONNXTraceRow};

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instruction: ONNXInstruction) -> Vec<ONNXInstruction> {
        let dummy_trace_row = ONNXTraceRow {
            instruction,
            layer_state: LayerState {
                input_vals: None,
                output_vals: None,
            },
        };
        Self::virtual_trace(dummy_trace_row)
            .into_iter()
            .map(|trace_row| trace_row.instruction)
            .collect()
    }
    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow>;
    fn sequence_output(x: i8, y: i8) -> i8;
}

pub mod max;
pub mod relu;
pub mod sigmoid;
pub mod div;
pub mod virtual_advice;
pub mod virtual_assert_valid_signed_remainder;
pub mod test;