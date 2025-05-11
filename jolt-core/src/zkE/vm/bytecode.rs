//! This module provides implementations reguarding the ByteCode secition of the Jolt proof.

use crate::{
    field::JoltField,
    jolt::{instruction::JoltInstructionSet, vm::bytecode::BytecodeRow},
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::collections::BTreeMap;
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
        // assert_eq!(virtual_address_map.insert((0, 0), 0), None);

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

fn witness() {}

#[cfg(test)]
mod tests {
    use super::preprocess;
    use crate::{
        jolt::vm::rv32i_vm::RV32I,
        zkE::{tests::testing_wasm_program, wasm_host::WASMProgram},
    };
    use ark_bn254::{Bn254, Fr};

    #[test]
    fn test_wasm_trace() {
        let wasm_program = testing_wasm_program();
        let (wasm_bytecode, init_memory) = wasm_program.decode();
        let pp = preprocess::<Fr, RV32I>(&wasm_bytecode);
        println!("Preprocessed bytecode: {pp:?}");
    }
}
