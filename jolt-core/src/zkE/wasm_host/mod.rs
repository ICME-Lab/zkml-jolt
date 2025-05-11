//! This module provides a wrapper around the `wasmi_tracer` library to trace
//! the execution of WASM programs and decode their bytecode.

use crate::jolt::vm::{bytecode::BytecodeRow, rv32i_vm::RV32I, JoltTraceStep};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs;
use tracer::ELFInstruction;
use wasmi_tracer::args::Args;

/// Represents a WASM program via its file path, function name, and inputs.
#[derive(Clone)]
pub struct WASMProgram {
    pub func: String,
    pub inputs: Vec<String>,
    pub file_path: String,
}

impl WASMProgram {
    /// Get the execution trace of the WASM program.
    ///
    /// # Returns
    ///
    /// * `Vec<JoltTraceStep<RV32I>>`: The execution trace of the WASM program.
    /// * `JoltWASMDevice`: The program i/o.
    pub fn trace(&self) -> Vec<JoltTraceStep<RV32I>> {
        let raw_trace = wasmi_tracer::trace(self.into()).unwrap();

        // Convert raw trace (Vec<RVTraceRow>) to JoltTraceStep
        let trace: Vec<_> = raw_trace
            .into_par_iter()
            .map(|row| {
                let instruction_lookup = RV32I::try_from(&row).ok();
                JoltTraceStep {
                    instruction_lookup,
                    bytecode_row: BytecodeRow::from_instruction::<RV32I>(&row.instruction),
                    memory_ops: (&row).into(),
                    circuit_flags: row.instruction.to_circuit_flags(),
                }
            })
            .collect();
        trace
    }

    /// Decodes the WASM bytecode and returns the decoded instructions and initial memory state.
    ///
    /// First converts [`PathBuf`] to &[u8] and then passes it to [`wasmi_tracer::decode`].
    ///
    /// # Returns
    /// * `Vec<ELFInstruction>`: The decoded instructions (bytecode).
    /// * `Vec<(u64, u8)>`: Memory addr, val tuples for the bytecode. i.e. the initial memory state of the bytecode.
    pub fn decode(&self) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
        let wasm_bytecode = fs::read(&self.file_path).unwrap();
        wasmi_tracer::decode(&wasm_bytecode)
    }
}

impl From<&WASMProgram> for Args {
    fn from(value: &WASMProgram) -> Self {
        Args::new(&value.file_path, &value.func, value.inputs.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::zkE::wasm_host::WASMProgram;

    #[test]
    fn test_wasm_trace() {
        let wasm_program = testing_wasm_program();
        let (_wasm_bytecode, _init_memory) = wasm_program.decode();
        let trace = wasm_program.trace();
        println!("WASM Trace: {trace:#?}");
    }
}
