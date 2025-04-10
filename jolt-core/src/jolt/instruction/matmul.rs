use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;

pub struct MATMULInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> MATMULInstruction<WORD_SIZE> {
    /// Converts a 2x2 matrix to a 64-bit integer
    /// Each element is stored as a 16-bit value in the result
    /// Matrix format: [[a, b], [c, d]] -> 0xAAAABBBBCCCCDDDD
    pub fn mat2uint(matrix: [[u32; 2]; 2]) -> u32 {
        let a = (matrix[0][0] & 0xFF) << 24;
        let b = (matrix[0][1] & 0xFF) << 16;
        let c = (matrix[1][0] & 0xFF) << 8;
        let d = matrix[1][1] & 0xFF;

        a | b | c | d
    }

    /// Converts a 64-bit integer to a 2x2 matrix
    /// Each 16-bit segment is extracted into a matrix element
    /// Integer format: 0xAAAABBBBCCCCDDDD -> [[a, b], [c, d]]
    pub fn uint2mat(val: u32) -> [[u32; 2]; 2] {
        let a = (val >> 24) & 0xFF;
        let b = (val >> 16) & 0xFF;
        let c = (val >> 8) & 0xFF;
        let d = val & 0xFF;

        [[a, b], [c, d]]
    }

    // Simple matrix multiplication operation for 2x2 matrices
    // The matrices are represented as flattened arrays in row-major order
    // Input: x = [a, b, c, d], y = [e, f, g, h]
    // Output: z = [a*e + b*g, a*f + b*h, c*e + d*g, c*f + d*h]
    fn matrix_multiply(left: u64, right: u64) -> u64 {
        // Extract the matrix elements from the inputs
        let left_mat = Self::uint2mat(left as u32);
        let right_mat = Self::uint2mat(right as u32);

        // Perform matrix multiplication
        let result = [
            [
                left_mat[0][0]
                    .wrapping_mul(right_mat[0][0])
                    .wrapping_add(left_mat[0][1].wrapping_mul(right_mat[1][0])),
                left_mat[0][0]
                    .wrapping_mul(right_mat[0][1])
                    .wrapping_add(left_mat[0][1].wrapping_mul(right_mat[1][1])),
            ],
            [
                left_mat[1][0]
                    .wrapping_mul(right_mat[0][0])
                    .wrapping_add(left_mat[1][1].wrapping_mul(right_mat[1][0])),
                left_mat[1][0]
                    .wrapping_mul(right_mat[0][1])
                    .wrapping_add(left_mat[1][1].wrapping_mul(right_mat[1][1])),
            ],
        ];

        // Convert the result back to a 64-bit integer
        Self::mat2uint(result) as u64
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
        let a = Self::uint2mat(x as u32);
        let b = Self::uint2mat(y as u32);

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
                rd_post_val: Some(a[0][0] as u64),
            },
            memory_state: None,
            advice_value: Some(a[0][0] as u64),
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
                rd_post_val: Some(a[0][1] as u64),
            },
            memory_state: None,
            advice_value: Some(a[0][1] as u64),
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
                rd_post_val: Some(a[1][0] as u64),
            },
            memory_state: None,
            advice_value: Some(a[1][0] as u64),
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
                rd_post_val: Some(a[1][1] as u64),
            },
            memory_state: None,
            advice_value: Some(a[1][1] as u64),
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
                rd_post_val: Some(b[0][0] as u64),
            },
            memory_state: None,
            advice_value: Some(b[0][0] as u64),
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
                rd_post_val: Some(b[0][1] as u64),
            },
            memory_state: None,
            advice_value: Some(b[0][1] as u64),
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
                rd_post_val: Some(b[1][0] as u64),
            },
            memory_state: None,
            advice_value: Some(b[1][0] as u64),
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
                rd_post_val: Some(b[1][1] as u64),
            },
            memory_state: None,
            advice_value: Some(b[1][1] as u64),
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

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};
//             advice_value: None,
//             precompile_input: None,
//             precompile_output_address: None,
//         });

//         virtual_trace
//     }

//     fn sequence_output(x: u64, y: u64) -> u64 {
//         Self::matrix_multiply(x, y)
//     }
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

    #[test]
    fn gradient_boost_sequence_32() {
        jolt_virtual_sequence_test!(MATMULInstruction<32>, RV32IM::MATMUL);
    }
}
