use tracer::ELFInstruction;

use crate::{
    field::JoltField,
    jolt::{
        instruction::JoltInstructionSet,
        vm::bytecode::{BytecodePreprocessing, BytecodeRow},
    },
};

fn preprocess<F, InstructionSet>(bytecode: &[ELFInstruction]) -> BytecodePreprocessing<F>
where
    F: JoltField,
    InstructionSet: JoltInstructionSet,
{
    let bytecode_rows: Vec<BytecodeRow> = bytecode
        .iter()
        .map(|instruction| BytecodeRow::from_instruction::<InstructionSet>(instruction))
        .collect();
    BytecodePreprocessing::<F>::preprocess(bytecode_rows)
}

fn witness() {}

#[cfg(test)]
mod tests {
    use super::preprocess;
    use crate::{jolt::vm::rv32i_vm::RV32I, zkE::wasm_host::WASMProgram};
    use ark_bn254::{Bn254, Fr};

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

        let (wasm_bytecode, init_memory) = wasm_program.decode();
        let pp = preprocess::<Fr, RV32I>(&wasm_bytecode);
        println!("Preprocessed bytecode: {pp:?}");
    }
}
