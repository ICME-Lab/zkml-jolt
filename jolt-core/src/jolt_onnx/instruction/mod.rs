//! This module provides the custom jolt instructions for the ONNX runtime.

use crate::{field::JoltField, jolt_onnx::tracer::tensor::QuantizedTensor};

pub mod max;
pub mod relu;
pub mod sigmoid;

pub trait JoltONNXInstruction {
    fn combine_instruction_results(&self, results: &[u64]) -> QuantizedTensor;
    fn inner_instructions(&self) -> Vec<JoltInstruction>;
}