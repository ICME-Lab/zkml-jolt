/// Its a virtual instruction for matrix multiplication
/// It takes two 2x2 matrices as input and returns a single 64-bit integer
/// The matrices are represented as 32-bit integers, where each 8-bit segment
/// Actually we want to pack the each element of the matrix into a single 32/64-bit integer
/// This will be solved after precompile supports
use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{add::ADDInstruction, mulu::MULUInstruction, JoltInstruction};

pub struct MATMULInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> MATMULInstruction<WORD_SIZE> {
    /// Converts a 2x2 matrix to a 32-bit integer
    /// Each element is stored as a 8-bit value in the result
    /// Matrix format: [[a, b], [c, d]] -> 0xAABBCCDD
    pub fn mat2uint(matrix: [[u32; 2]; 2]) -> u32 {
        let a = (matrix[0][0] & 0xFF) << 24;
        let b = (matrix[0][1] & 0xFF) << 16;
        let c = (matrix[1][0] & 0xFF) << 8;
        let d = matrix[1][1] & 0xFF;

        a | b | c | d
    }

    /// Converts a 32-bit integer to a 2x2 matrix
    /// Each 8-bit segment is extracted into a matrix element
    /// Integer format: 0xAABBCCDD -> [[a, b], [c, d]]
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
    // 1. Extract the elements from input matrices (8 registers)
    // 2. Perform multiplications (8 operations)
    // 3. Perform additions (4 operations)
    // 4. Create the final packed result (1 operation)
    // 5. Move the result to the destination register (1 operation)
    const SEQUENCE_LENGTH: usize = 22;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::MATMUL);

        // Input matrices
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();

        // Create registers for matrix elements and intermediate results
        let r_a00 = Some(virtual_register_index(0)); // left matrix elements
        let r_a01 = Some(virtual_register_index(1));
        let r_a10 = Some(virtual_register_index(2));
        let r_a11 = Some(virtual_register_index(3));

        let r_b00 = Some(virtual_register_index(4)); // right matrix elements
        let r_b01 = Some(virtual_register_index(5));
        let r_b10 = Some(virtual_register_index(6));
        let r_b11 = Some(virtual_register_index(7));

        let r_mul1 = Some(virtual_register_index(8)); // Intermediate multiplication results
        let r_mul2 = Some(virtual_register_index(9));
        let r_mul3 = Some(virtual_register_index(10));
        let r_mul4 = Some(virtual_register_index(11));
        let r_mul5 = Some(virtual_register_index(12));
        let r_mul6 = Some(virtual_register_index(13));
        let r_mul7 = Some(virtual_register_index(14));
        let r_mul8 = Some(virtual_register_index(15));

        let r_add1 = Some(virtual_register_index(16)); // Intermediate addition results
        let r_add2 = Some(virtual_register_index(17));
        let r_add3 = Some(virtual_register_index(18));
        let r_add4 = Some(virtual_register_index(19));

        let r_result = Some(virtual_register_index(20)); // Final result

        let mut virtual_trace: Vec<RVTraceRow> = vec![];

        // Extract elements from matrices
        let a = Self::uint2mat(x as u32);
        let b = Self::uint2mat(y as u32);

        // Calculate ahead of time for validation
        let _expected_result = Self::matrix_multiply(x, y);

        // Step 1: Store left matrix elements into virtual registers
        let a00_val = a[0][0] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_a00,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a00_val),
            },
            memory_state: None,
            advice_value: Some(a00_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let a01_val = a[0][1] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_a01,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a01_val),
            },
            memory_state: None,
            advice_value: Some(a01_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let a10_val = a[1][0] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_a10,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a10_val),
            },
            memory_state: None,
            advice_value: Some(a10_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let a11_val = a[1][1] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_a11,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(a11_val),
            },
            memory_state: None,
            advice_value: Some(a11_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 2: Store right matrix elements into virtual registers
        let b00_val = b[0][0] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_b00,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(b00_val),
            },
            memory_state: None,
            advice_value: Some(b00_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let b01_val = b[0][1] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_b01,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(b01_val),
            },
            memory_state: None,
            advice_value: Some(b01_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let b10_val = b[1][0] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_b10,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(b10_val),
            },
            memory_state: None,
            advice_value: Some(b10_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        let b11_val = b[1][1] as u64;
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_b11,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(b11_val),
            },
            memory_state: None,
            advice_value: Some(b11_val),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 3: Perform multiplications using MULU
        // c00 = a00*b00 + a01*b10
        let mul1_val = MULUInstruction::<WORD_SIZE>(a00_val, b00_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a00,
                rs2: r_b00,
                rd: r_mul1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a00_val),
                rs2_val: Some(b00_val),
                rd_post_val: Some(mul1_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let mul2_val = MULUInstruction::<WORD_SIZE>(a01_val, b10_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a01,
                rs2: r_b10,
                rd: r_mul2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a01_val),
                rs2_val: Some(b10_val),
                rd_post_val: Some(mul2_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c01 = a00*b01 + a01*b11
        let mul3_val = MULUInstruction::<WORD_SIZE>(a00_val, b01_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a00,
                rs2: r_b01,
                rd: r_mul3,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a00_val),
                rs2_val: Some(b01_val),
                rd_post_val: Some(mul3_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let mul4_val = MULUInstruction::<WORD_SIZE>(a01_val, b11_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a01,
                rs2: r_b11,
                rd: r_mul4,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a01_val),
                rs2_val: Some(b11_val),
                rd_post_val: Some(mul4_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c10 = a10*b00 + a11*b10
        let mul5_val = MULUInstruction::<WORD_SIZE>(a10_val, b00_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a10,
                rs2: r_b00,
                rd: r_mul5,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a10_val),
                rs2_val: Some(b00_val),
                rd_post_val: Some(mul5_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let mul6_val = MULUInstruction::<WORD_SIZE>(a11_val, b10_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a11,
                rs2: r_b10,
                rd: r_mul6,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a11_val),
                rs2_val: Some(b10_val),
                rd_post_val: Some(mul6_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c11 = a10*b01 + a11*b11
        let mul7_val = MULUInstruction::<WORD_SIZE>(a10_val, b01_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a10,
                rs2: r_b01,
                rd: r_mul7,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a10_val),
                rs2_val: Some(b01_val),
                rd_post_val: Some(mul7_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let mul8_val = MULUInstruction::<WORD_SIZE>(a11_val, b11_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: r_a11,
                rs2: r_b11,
                rd: r_mul8,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(a11_val),
                rs2_val: Some(b11_val),
                rd_post_val: Some(mul8_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 4: Perform additions to compute final matrix elements
        // c00 = mul1 + mul2
        let add1_val = ADDInstruction::<WORD_SIZE>(mul1_val, mul2_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: r_mul1,
                rs2: r_mul2,
                rd: r_add1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(mul1_val),
                rs2_val: Some(mul2_val),
                rd_post_val: Some(add1_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c01 = mul3 + mul4
        let add2_val = ADDInstruction::<WORD_SIZE>(mul3_val, mul4_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: r_mul3,
                rs2: r_mul4,
                rd: r_add2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(mul3_val),
                rs2_val: Some(mul4_val),
                rd_post_val: Some(add2_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c10 = mul5 + mul6
        let add3_val = ADDInstruction::<WORD_SIZE>(mul5_val, mul6_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: r_mul5,
                rs2: r_mul6,
                rd: r_add3,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(mul5_val),
                rs2_val: Some(mul6_val),
                rd_post_val: Some(add3_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // c11 = mul7 + mul8
        let add4_val = ADDInstruction::<WORD_SIZE>(mul7_val, mul8_val).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: r_mul7,
                rs2: r_mul8,
                rd: r_add4,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(mul7_val),
                rs2_val: Some(mul8_val),
                rd_post_val: Some(add4_val),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 5: Create the final result - pack the computed values into a single integer
        // This is a more complex operation, so we'll use ADVICE to guide it
        let result_matrix = [
            [add1_val as u32 & 0xFF, add2_val as u32 & 0xFF],
            [add3_val as u32 & 0xFF, add4_val as u32 & 0xFF],
        ];
        let packed_result = Self::mat2uint(result_matrix) as u64;
        let exact_result = Self::matrix_multiply(x, y);
        assert_eq!(packed_result, exact_result);

        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_ADVICE,
                rs1: None,
                rs2: None,
                rd: r_result,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: None,
                rs2_val: None,
                rd_post_val: Some(packed_result),
            },
            memory_state: None,
            advice_value: Some(packed_result),
            precompile_input: None,
            precompile_output_address: None,
        });

        // Step 6: Move the result to the destination register
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVE,
                rs1: r_result,
                rs2: None,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - 1 - virtual_trace.len()),
            },
            register_state: RegisterState {
                rs1_val: Some(packed_result),
                rs2_val: None,
                rd_post_val: Some(packed_result),
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

    #[test]
    fn test_mat2uint() {
        let matrix: [[u32; 2]; 2] = [[100, 200], [220, 250]];
        let packed = MATMULInstruction::<32>::mat2uint(matrix);
        let unpacked = MATMULInstruction::<32>::uint2mat(packed);
        assert_eq!(matrix, unpacked);
    }
}
