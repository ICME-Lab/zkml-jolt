//! This module provides implementations reguarding the ByteCode secition of the Jolt proof.
use crate::lasso::memory_checking::ExogenousOpenings;
use crate::poly::compact_polynomial::SmallScalar;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::{
    field::JoltField,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            bytecode::{
                BytecodeCommitments, BytecodeOpenings, BytecodePolynomials, BytecodeRow,
                BytecodeStuff,
            },
            JoltPolynomials, JoltTraceStep,
        },
    },
    lasso::memory_checking::{
        MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier, MultisetHashes,
        NoExogenousOpenings, StructuredPolynomialData,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme, compact_polynomial::CompactPolynomial,
        identity_poly::IdentityPolynomial, multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::grand_product::BatchedGrandProductProof,
    utils::transcript::Transcript,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::*;
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::HashSet;
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

// HACK
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct WASMMemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
    Openings: StructuredPolynomialData<F> + Sync + CanonicalSerialize + CanonicalDeserialize,
    OtherOpenings: ExogenousOpenings<F> + Sync,
{
    /// Read/write/init/final multiset hashes for each memory
    pub multiset_hashes: MultisetHashes<F>,
    /// The read and write grand products for every memory has the same size,
    /// so they can be batched.
    pub read_write_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The init and final grand products for every memory has the same size,
    /// so they can be batched.
    pub init_final_grand_product: BatchedGrandProductProof<PCS, ProofTranscript>,
    /// The openings associated with the grand products.
    pub openings: Openings,
    pub exogenous_openings: OtherOpenings,
}

impl<F, PCS, Openings, OtherOpenings, ProofTranscript>
    From<MemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>>
    for WASMMemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    Openings: StructuredPolynomialData<F> + Sync + CanonicalSerialize + CanonicalDeserialize,
    OtherOpenings: ExogenousOpenings<F> + Sync,
    ProofTranscript: Transcript,
{
    fn from(proof: MemoryCheckingProof<F, PCS, Openings, OtherOpenings, ProofTranscript>) -> Self {
        Self {
            multiset_hashes: proof.multiset_hashes,
            read_write_grand_product: proof.read_write_grand_product,
            init_final_grand_product: proof.init_final_grand_product,
            openings: proof.openings,
            exogenous_openings: proof.exogenous_openings,
        }
    }
}

pub type WASMBytecodeProof<F, PCS, ProofTranscript> =
    WASMMemoryCheckingProof<F, PCS, BytecodeOpenings<F>, NoExogenousOpenings, ProofTranscript>;

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for WASMBytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Polynomials = BytecodePolynomials<F>;
    type Openings = BytecodeOpenings<F>;
    type Commitments = BytecodeCommitments<PCS, ProofTranscript>;
    type Preprocessing = WASMBytecodePreprocessing<F>;

    // [virtual_address, elf_address, opcode, rd, rs1, rs2, imm, t]
    type MemoryTuple = [F; 8];

    fn fingerprint(inputs: &Self::MemoryTuple, gamma: &F, tau: &F) -> F {
        let mut result = F::zero();
        let mut gamma_term = F::one();
        for input in inputs {
            result += *input * gamma_term;
            gamma_term *= *gamma;
        }
        result - *tau
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::compute_leaves")]
    fn compute_leaves(
        preprocessing: &WASMBytecodePreprocessing<F>,
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let num_ops = polynomials.a_read_write.len();
        let bytecode_size = preprocessing.v_init_final[0].len();

        let mut gamma_terms = [F::zero(); 7];
        let mut gamma_term = F::one();
        for i in 0..7 {
            gamma_term *= *gamma;
            gamma_terms[i] = gamma_term;
        }

        let a: &CompactPolynomial<u32, F> = (&polynomials.a_read_write).try_into().unwrap();
        let v_address: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[4]).try_into().unwrap();
        let v_imm: &CompactPolynomial<i64, F> = (&polynomials.v_read_write[5]).try_into().unwrap();
        let t: &CompactPolynomial<u32, F> = (&polynomials.t_read).try_into().unwrap();

        let read_leaves: Vec<F> = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + a[i].field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    + t[i].field_mul(gamma_terms[6])
                    - tau
            })
            .collect();

        // TODO(moodlezoup): Compute write_leaves from read_leaves
        let write_leaves: Vec<F> = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + a[i].field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    + (t[i] + 1).field_mul(gamma_terms[6])
                    - tau
            })
            .collect();

        let v_address: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[4]).try_into().unwrap();
        let v_imm: &CompactPolynomial<i64, F> =
            (&preprocessing.v_init_final[5]).try_into().unwrap();

        let init_leaves: Vec<F> = (0..bytecode_size)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + (i as u64).field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    // + gamma_terms[6] * 0
                    - tau
            })
            .collect();

        // TODO(moodlezoup): Compute final_leaves from init_leaves
        let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
        let final_leaves: Vec<F> = (0..bytecode_size)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + (i as u64).field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    + t_final[i].field_mul(gamma_terms[6])
                    - tau
            })
            .collect();

        // TODO(moodlezoup): avoid concat
        (
            ([read_leaves, write_leaves].concat(), 2),
            ([init_leaves, final_leaves].concat(), 2),
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Bytecode memory checking"
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for WASMBytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut BytecodeOpenings<F>,
        preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));

        openings.v_init_final = Some(
            MultilinearPolynomial::batch_evaluate(
                &preprocessing.v_init_final.iter().collect::<Vec<_>>(),
                r_init_final,
            )
            .0
            .try_into()
            .unwrap(),
        );
    }

    fn read_tuples(
        _: &WASMBytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.v_read_write[5], // imm
            openings.a_read_write,
            openings.v_read_write[0], // address
            openings.v_read_write[1], // opcode
            openings.v_read_write[2], // rd
            openings.v_read_write[3], // rs1
            openings.v_read_write[4], // rs2
            openings.t_read,
        ]]
    }
    fn write_tuples(
        _: &WASMBytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.v_read_write[5], // imm
            openings.a_read_write,
            openings.v_read_write[0], // address
            openings.v_read_write[1], // opcode
            openings.v_read_write[2], // rd
            openings.v_read_write[3], // rs1
            openings.v_read_write[4], // rs2
            openings.t_read + F::one(),
        ]]
    }
    fn init_tuples(
        _: &WASMBytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let v_init_final = openings.v_init_final.unwrap();
        vec![[
            v_init_final[5], // imm
            openings.a_init_final.unwrap(),
            v_init_final[0], // address
            v_init_final[1], // opcode
            v_init_final[2], // rd
            v_init_final[3], // rs1
            v_init_final[4], // rs2
            F::zero(),
        ]]
    }
    fn final_tuples(
        _: &WASMBytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let v_init_final = openings.v_init_final.unwrap();
        vec![[
            v_init_final[5], // imm
            openings.a_init_final.unwrap(),
            v_init_final[0], // address
            v_init_final[1], // opcode
            v_init_final[2], // rd
            v_init_final[3], // rs1
            v_init_final[4], // rs2
            openings.t_final,
        ]]
    }
}

impl<F, PCS, ProofTranscript> WASMBytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "BytecodePolynomials::validate_bytecode")]
    pub fn validate_bytecode(bytecode: &[BytecodeRow], trace: &[BytecodeRow]) {
        let mut bytecode_map: BTreeMap<usize, &BytecodeRow> = BTreeMap::new();

        for bytecode_row in bytecode.iter() {
            bytecode_map.insert(bytecode_row.address, bytecode_row);
        }

        for (i, trace_row) in trace.iter().enumerate() {
            let expected = *bytecode_map
                .get(&trace_row.address)
                .expect("couldn't find in bytecode");
            if *expected != *trace_row {
                panic!("Mismatch at index {i}: expected {expected:?}, got {trace_row:?}",);
            }
        }
    }

    #[tracing::instrument(skip_all, name = "WASMMemoryCheckingProof::generate_witness")]
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
    use crate::utils::transcript::KeccakTranscript;
    use crate::{jolt::vm::rv32i_vm::RV32I, zkE::tests::add_sub_mul_wasm_program};
    use ark_bn254::{Bn254, Fr};
    use itertools::Itertools;

    #[test]
    fn test_wasm_bytecode() {
        let wasm_program = add_sub_mul_wasm_program();
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

        let _witness = WASMBytecodeProof::<Fr, HyperKZG<Bn254, _>, KeccakTranscript>::wasm_witness::<
            RV32I,
        >(&pp, &mut execution_trace);
    }
}
