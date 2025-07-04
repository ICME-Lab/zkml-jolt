//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::{field::JoltField, jolt::instruction::JoltInstruction, jolt_onnx::tracer::tensor::QuantizedTensor};

pub mod max;
pub mod relu;
pub mod sigmoid;

/// Trait for ONNX instructions.
pub trait JoltONNXInstruction<InnerInstruction: JoltInstruction> {
    /// Combine the results of the inner instructions into a quantized tensor.
    fn combine_instruction_results(&self, results: &[u32]) -> QuantizedTensor;

    /// Retrieve the inner instructions.
    fn inner_instructions(&self) -> Vec<InnerInstruction>;
}