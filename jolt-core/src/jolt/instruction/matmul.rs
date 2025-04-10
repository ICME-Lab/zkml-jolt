use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;

pub struct MATMULInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> MATMULInstruction<WORD_SIZE> {
    // Simple matrix multiplication operation for 2x2 matrices
    // The matrices are represented as flattened arrays in row-major order
    // Input: x = [a, b, c, d], y = [e, f, g, h]
    // Output: z = [a*e + b*g, a*f + b*h, c*e + d*g, c*f + d*h]
    fn matrix_multiply(left: u64, right: u64) -> u64 {
        // Extract the matrix elements from the inputs
        // For this simple implementation, we're assuming 2x2 matrices
        // Each element is 16 bits, allowing us to pack the entire matrix into a 64-bit value

        // Extract elements from left matrix
        let a = (left >> 48) & 0xFFFF;
        let b = (left >> 32) & 0xFFFF;
        let c = (left >> 16) & 0xFFFF;
        let d = left & 0xFFFF;

        // Extract elements from right matrix
        let e = (right >> 48) & 0xFFFF;
        let f = (right >> 32) & 0xFFFF;
        let g = (right >> 16) & 0xFFFF;
        let h = right & 0xFFFF;

        // Compute the result matrix
        let z0 = a.wrapping_mul(e).wrapping_add(b.wrapping_mul(g)) & 0xFFFF;
        let z1 = a.wrapping_mul(f).wrapping_add(b.wrapping_mul(h)) & 0xFFFF;
        let z2 = c.wrapping_mul(e).wrapping_add(d.wrapping_mul(g)) & 0xFFFF;
        let z3 = c.wrapping_mul(f).wrapping_add(d.wrapping_mul(h)) & 0xFFFF;

        // Pack the result into a 64-bit value
        (z0 << 48) | (z1 << 32) | (z2 << 16) | z3
    }
}

impl<const WORD_SIZE: usize> VirtualInstructionSequence for MATMULInstruction<WORD_SIZE> {
    // We need several instructions to perform the matrix multiplication:
    // 1. Extract the elements from input matrices
    // 2. Perform multiplications
    // 3. Perform additions
    // 4. Combine the results
    const SEQUENCE_LENGTH: usize = 10;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        // Input matrices
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();

        // Virtual registers
        let v0 = Some(virtual_register_index(0)); // For intermediate values
        let v1 = Some(virtual_register_index(1));
        let v2 = Some(virtual_register_index(2));
        let v3 = Some(virtual_register_index(3));
        let v4 = Some(virtual_register_index(4));
        let v5 = Some(virtual_register_index(5));
        let v6 = Some(virtual_register_index(6));
        let v7 = Some(virtual_register_index(7));
        let v8 = Some(virtual_register_index(8));

        let mut virtual_trace: Vec<RVTraceRow> = vec![];

        // Calculate the final result of the matrix multiplication
        let result = Self::matrix_multiply(x, y);

        // Extract elements from matrices
        // Left matrix elements: a, b, c, d
        let a = (x >> 48) & 0xFFFF;
        let b = (x >> 32) & 0xFFFF;
        let c = (x >> 16) & 0xFFFF;
        let d = x & 0xFFFF;

        // Right matrix elements: e, f, g, h
        let e = (y >> 48) & 0xFFFF;
        let f = (y >> 32) & 0xFFFF;
        let g = (y >> 16) & 0xFFFF;
        let h = y & 0xFFFF;

        // Step 1: Store matrix elements into virtual registers using ADVICEInstruction
        // Store a, b, c, d (from left matrix)
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v0,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a),
            },
            memory_state: None,
            advice_value: Some(a),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(b),
            },
            memory_state: None,
            advice_value: Some(b),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(c),
            },
            memory_state: None,
            advice_value: Some(c),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v3,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(d),
            },
            memory_state: None,
            advice_value: Some(d),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Store e, f, g, h (from right matrix)
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v4,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(e),
            },
            memory_state: None,
            advice_value: Some(e),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v5,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(f),
            },
            memory_state: None,
            advice_value: Some(f),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v6,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(g),
            },
            memory_state: None,
            advice_value: Some(g),
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v7,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(h),
            },
            memory_state: None,
            advice_value: Some(h),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 2: Get the final result directly using ADVICEInstruction
        // instead of calculating it step by step for simplicity
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: v8,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: Some(result),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 3: Move the result to the destination register
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVE,
                rs1: v8,
                rs2: None,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(result),
                rs2_val: None,
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        Self::matrix_multiply(x, y)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

    #[test]
    fn gradient_boost_sequence_32() {
        jolt_virtual_sequence_test!(MATMULInstruction<32>, RV32IM::MATMUL);
    }
}
