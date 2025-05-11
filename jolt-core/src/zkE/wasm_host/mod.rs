use std::fs;
use std::fs::File;

use crate::jolt::vm::{bytecode::BytecodeRow, rv32i_vm::RV32I, JoltTraceStep};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use tracer::ELFInstruction;
use wasmi_tracer::args::Args;

#[derive(Clone)]
pub struct WASMProgram {
    pub func: String,
    pub inputs: Vec<String>,
    pub file_path: String,
}

impl WASMProgram {
    pub fn trace(&self) -> Vec<JoltTraceStep<RV32I>> {
        let raw_trace = wasmi_tracer::trace(self.into()).unwrap();
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
        // WASM inputs
        let stake = "1500".to_string(); // Amount of LP tokens or liquidity staked by the user.
        let duration_boost = "3".to_string(); // Boost multiplier based on how long the stake was held (e.g., 3 = 3 months).
        let volume_boost = "2".to_string(); // Additional multiplier based on trading volume in the pool during the staking period.
        let penalty = "500".to_string(); // Penalty applied for early withdrawal or performance issues (e.g., protocol downgrade).

        let wasm_program = WASMProgram {
            func: "main".to_string(),
            inputs: vec![stake, duration_boost, volume_boost, penalty],
            file_path: "./wasms/add_sub_mul_32.wat".to_string(),
        };

        let (_wasm_bytecode, _init_memory) = wasm_program.decode();
        let trace = wasm_program.trace();
        println!("WASM Trace: {trace:#?}");
    }
}
