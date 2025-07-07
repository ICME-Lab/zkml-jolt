//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow};
use crate::{field::JoltField, jolt::instruction::JoltInstruction, jolt_onnx::tracer::tensor::QuantizedTensor};

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;
    fn virtual_sequence(instruction: ONNXInstruction) -> Vec<ONNXInstruction> {
        let dummy_trace_row = ONNXTraceRow {
            instruction,
            layer_state: LayerState {
                input_vals: vec![],
                output_vals: vec![],
            },
            advice_value: vec![],
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
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod test;

/// Trait for ONNX instructions.
pub trait JoltONNXInstruction<InnerInstruction: JoltInstruction> {
    /// Combine the results of the inner instructions into a quantized tensor.
    fn combine_instruction_results(&self, results: &[u32]) -> QuantizedTensor;

    /// Retrieve the inner instructions.
    fn inner_instructions(&self) -> Vec<InnerInstruction>;
}
