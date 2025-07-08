//! This module provides the custom jolt instructions for the ONNX runtime.

use strum::{EnumCount, IntoEnumIterator};

use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow};
use crate::jolt::instruction::JoltInstruction;

pub mod max;
pub mod relu;
pub mod sigmoid;
pub mod div;
pub mod test;

/// Trait for the virtual instruction sequence.
pub trait VirtualInstructionSequence {
    /// The length of the virtual instruction sequence.
    const SEQUENCE_LENGTH: usize;
    /// Returns the virtual instruction sequence for the given instruction.
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
    /// Returns the virtual trace for the given instruction.        
    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow>;
    /// Returns the output of the instruction for the given input.
    fn sequence_output(x: i8, y: i8) -> i8;
}


/// Trait for the Jolt ONNX instruction set.
pub trait JoltONNXInstructionSet:
    JoltInstruction + IntoEnumIterator + EnumCount + for<'a> TryFrom<&'a ONNXInstruction> + Send + Sync
{
    /// Returns the index of the instruction in the enum.
    fn enum_index(instruction: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(instruction as *const Self as *const u8) };
        byte as usize
    }
}


