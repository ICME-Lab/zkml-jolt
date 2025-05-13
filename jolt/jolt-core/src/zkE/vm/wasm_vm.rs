use crate::{
    field::JoltField,
    jolt::vm::rv32i_vm::{RV32ISubtables, RV32I},
    poly::commitment::commitment_scheme::CommitmentScheme,
    r1cs::constraints::JoltRV32IMConstraints,
    utils::transcript::Transcript,
};

use super::JoltWASM;

pub enum WASMJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, PCS, ProofTranscript> JoltWASM<F, PCS, C, M, ProofTranscript> for WASMJoltVM
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;
    type Constraints = JoltRV32IMConstraints;
}

#[cfg(test)]
mod tests {
    use super::{WASMJoltVM, C};
    use crate::utils::transcript::KeccakTranscript;
    use crate::zkE::tests::{add_sub_mul_wasm_program, bitwise_arith_wasm_program};
    use crate::zkE::vm::JoltWASM;
    use crate::zkE::wasm_host::WASMProgram;
    use crate::{poly::commitment::hyperkzg::HyperKZG, zkE::vm::JoltProverPreprocessing};
    use ark_bn254::{Bn254, Fr};

    fn test_wasm_e2e_with(wasm_program: WASMProgram) {
        let (wasm_bytecode, _init_memory) = wasm_program.decode();

        // Preprocessing
        let preprocessing: JoltProverPreprocessing<C, Fr, HyperKZG<Bn254, _>, KeccakTranscript> =
            WASMJoltVM::prover_preprocess(wasm_bytecode.clone(), 1 << 20, 1 << 20);

        // Prove
        let (execution_trace, program_io) = wasm_program.trace();
        let (snark, commitments, _, _debug_info) =
            WASMJoltVM::prove(program_io.clone(), execution_trace, preprocessing.clone());

        // Verify
        WASMJoltVM::verify(preprocessing.shared, snark, commitments, program_io, None).unwrap();
    }

    #[test]
    fn test_add_sub_mul() {
        let wasm_program = add_sub_mul_wasm_program();
        test_wasm_e2e_with(wasm_program);
    }

    #[test]
    fn test_bitwise_arith() {
        let wasm_program = bitwise_arith_wasm_program();
        test_wasm_e2e_with(wasm_program);
    }
}
