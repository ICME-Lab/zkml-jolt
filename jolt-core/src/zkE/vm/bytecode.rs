//! This module provides implementations reguarding the ByteCode secition of the Jolt proof.

use crate::{
    field::JoltField,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            bytecode::{BytecodeOpenings, BytecodePolynomials, BytecodeRow, BytecodeStuff},
            JoltTraceStep,
        },
    },
    lasso::memory_checking::{MemoryCheckingProof, NoExogenousOpenings},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    utils::transcript::Transcript,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::collections::{BTreeMap, HashSet};
use tracer::ELFInstruction;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct WASMBytecodePreprocessing<F: JoltField> {
    /// Size of the (padded) bytecode.
    code_size: usize,
    /// MLE of init/final values. Bytecode is read-only data, so the final memory values are unchanged from
    /// the initial memory values. There are six values (address, bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_init_final` comprises six polynomials.
    v_init_final: [MultilinearPolynomial<F>; 6],
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl<F: JoltField> WASMBytecodePreprocessing<F> {
    #[tracing::instrument(skip_all, name = "WASMBytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<BytecodeRow>) -> Self {
        let mut virtual_address_map: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode

        // TODO: Check we don't need WASM pc to be above address 0x8000_0000
        for instruction in bytecode.iter_mut() {
            assert_eq!(
                virtual_address_map.insert(
                    (
                        instruction.address,
                        instruction.virtual_sequence_remaining.unwrap_or(0)
                    ),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, BytecodeRow::no_op(0));
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        let code_size = bytecode.len().next_power_of_two();
        bytecode.resize(code_size, BytecodeRow::no_op(0));

        let mut address = vec![];
        let mut bitflags = vec![];
        let mut rd = vec![];
        let mut rs1 = vec![];
        let mut rs2 = vec![];
        let mut imm = vec![];

        for instruction in bytecode {
            address.push(instruction.address as u64);
            bitflags.push(instruction.bitflags);
            rd.push(instruction.rd);
            rs1.push(instruction.rs1);
            rs2.push(instruction.rs2);
            imm.push(instruction.imm);
        }

        let v_init_final = [
            MultilinearPolynomial::from(address),
            MultilinearPolynomial::from(bitflags),
            MultilinearPolynomial::from(rd),
            MultilinearPolynomial::from(rs1),
            MultilinearPolynomial::from(rs2),
            MultilinearPolynomial::from(imm),
        ];
        Self {
            v_init_final,
            code_size,
            virtual_address_map,
        }
    }
}

fn preprocess<F, InstructionSet>(bytecode: &[ELFInstruction]) -> WASMBytecodePreprocessing<F>
where
    F: JoltField,
    InstructionSet: JoltInstructionSet,
{
    let bytecode_rows: Vec<BytecodeRow> = bytecode
        .iter()
        .map(|instruction| BytecodeRow::from_instruction::<InstructionSet>(instruction))
        .collect();
    WASMBytecodePreprocessing::<F>::preprocess(bytecode_rows)
}

pub type WASMBytecodeProof<F, PCS, ProofTranscript> =
    MemoryCheckingProof<F, PCS, BytecodeOpenings<F>, NoExogenousOpenings, ProofTranscript>;

impl<F, PCS, ProofTranscript> WASMBytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "WASMBytecodeProof::generate_witness")]
    pub fn wasm_witness<InstructionSet: JoltInstructionSet>(
        preprocessing: &WASMBytecodePreprocessing<F>,
        trace: &mut Vec<JoltTraceStep<InstructionSet>>,
    ) -> BytecodePolynomials<F> {
        let num_ops = trace.len();

        let mut a_read_write: Vec<u32> = vec![0; num_ops];
        let mut read_cts: Vec<u32> = vec![0; num_ops];
        let mut final_cts: Vec<u32> = vec![0; preprocessing.code_size];

        for (step_index, step) in trace.iter_mut().enumerate() {
            // TODO: Figure out exact reason for this compression
            // if !step.bytecode_row.address.is_zero() {
            //     assert!(step.bytecode_row.address >= RAM_START_ADDRESS as usize);
            //     assert!(step.bytecode_row.address % BYTES_PER_INSTRUCTION == 0);
            //     // Compress instruction address for more efficient commitment:
            //     step.bytecode_row.address = 1
            //         + (step.bytecode_row.address - RAM_START_ADDRESS as usize)
            //             / BYTES_PER_INSTRUCTION;
            // }

            let virtual_address = preprocessing
                .virtual_address_map
                .get(&(
                    step.bytecode_row.address,
                    step.bytecode_row.virtual_sequence_remaining.unwrap_or(0),
                ))
                .unwrap();
            a_read_write[step_index] = *virtual_address as u32;
            let counter = final_cts[*virtual_address];
            read_cts[step_index] = counter;
            final_cts[*virtual_address] = counter + 1;
        }

        let mut address = vec![];
        let mut bitflags = vec![];
        let mut rd = vec![];
        let mut rs1 = vec![];
        let mut rs2 = vec![];
        let mut imm = vec![];

        for step in trace {
            address.push(step.bytecode_row.address as u64);
            bitflags.push(step.bytecode_row.bitflags);
            rd.push(step.bytecode_row.rd);
            rs1.push(step.bytecode_row.rs1);
            rs2.push(step.bytecode_row.rs2);
            imm.push(step.bytecode_row.imm);
        }

        let v_read_write = [
            MultilinearPolynomial::from(address),
            MultilinearPolynomial::from(bitflags),
            MultilinearPolynomial::from(rd),
            MultilinearPolynomial::from(rs1),
            MultilinearPolynomial::from(rs2),
            MultilinearPolynomial::from(imm),
        ];
        let t_read: MultilinearPolynomial<F> = MultilinearPolynomial::from(read_cts);
        let t_final = MultilinearPolynomial::from(final_cts);

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut final_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for a in 0..t_final.len() {
            let t: F = t_final.get_coeff(a);
            init_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0].get_coeff(a),
                    preprocessing.v_init_final[1].get_coeff(a),
                    preprocessing.v_init_final[2].get_coeff(a),
                    preprocessing.v_init_final[3].get_coeff(a),
                    preprocessing.v_init_final[4].get_coeff(a),
                    preprocessing.v_init_final[5].get_coeff(a),
                ],
                0,
            ));
            final_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0].get_coeff(a),
                    preprocessing.v_init_final[1].get_coeff(a),
                    preprocessing.v_init_final[2].get_coeff(a),
                    preprocessing.v_init_final[3].get_coeff(a),
                    preprocessing.v_init_final[4].get_coeff(a),
                    preprocessing.v_init_final[5].get_coeff(a),
                ],
                t.to_u64().unwrap(),
            ));
        }

        #[cfg(test)]
        let mut read_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut write_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for (i, a) in a_read_write.iter().enumerate() {
            read_tuples.insert((
                *a as u64,
                [
                    v_read_write[0].get_coeff(i),
                    v_read_write[1].get_coeff(i),
                    v_read_write[2].get_coeff(i),
                    v_read_write[3].get_coeff(i),
                    v_read_write[4].get_coeff(i),
                    v_read_write[5].get_coeff(i),
                ],
                t_read.get_coeff(i).to_u64().unwrap(),
            ));
            write_tuples.insert((
                *a as u64,
                [
                    v_read_write[0].get_coeff(i),
                    v_read_write[1].get_coeff(i),
                    v_read_write[2].get_coeff(i),
                    v_read_write[3].get_coeff(i),
                    v_read_write[4].get_coeff(i),
                    v_read_write[5].get_coeff(i),
                ],
                t_read.get_coeff(i).to_u64().unwrap() + 1,
            ));
        }

        #[cfg(test)]
        {
            let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
            let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
            let set_difference: Vec<_> = init_write.symmetric_difference(&read_final).collect();
            assert_eq!(set_difference.len(), 0);
        }

        let a_read_write = MultilinearPolynomial::from(a_read_write);

        BytecodeStuff {
            a_read_write,
            v_read_write,
            t_read,
            t_final,
            a_init_final: None,
            v_init_final: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{preprocess, WASMBytecodeProof};
    use crate::jolt::vm::bytecode::BytecodeRow;
    use crate::jolt::vm::JoltTraceStep;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use crate::{
        jolt::vm::rv32i_vm::RV32I,
        zkE::{tests::testing_wasm_program, wasm_host::WASMProgram},
    };
    use ark_bn254::{Bn254, Fr};
    use itertools::Itertools;

    #[test]
    fn test_wasm_bytecode() {
        let wasm_program = testing_wasm_program();
        let (wasm_bytecode, _init_memory) = wasm_program.decode();
        let pp = preprocess::<Fr, RV32I>(&wasm_bytecode);

        // Pad the trace & generate the witness polynomials
        let (mut execution_trace, _program_io) = wasm_program.trace();
        JoltTraceStep::pad(&mut execution_trace);

        // Get the bytecode trace & validate the bytecode
        let mut wasm_bytecode_trace: Vec<BytecodeRow> = wasm_bytecode
            .iter()
            .map(BytecodeRow::from_instruction::<RV32I>)
            .collect();
        wasm_bytecode_trace.insert(0, BytecodeRow::no_op(0));
        WASMBytecodeProof::<Fr, HyperKZG<Bn254, _>, KeccakTranscript>::validate_bytecode(
            &wasm_bytecode_trace,
            &execution_trace
                .iter()
                .map(|step| step.bytecode_row.clone())
                .collect_vec(),
        );

        let witness = WASMBytecodeProof::<Fr, HyperKZG<Bn254, _>, KeccakTranscript>::wasm_witness::<
            RV32I,
        >(&pp, &mut execution_trace);
    }
}
