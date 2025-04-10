#![allow(unused_assignments, asm_sub_register)]

use onnx_util::{ComputationalGraph, OperationType};

#[jolt::provable]
fn execute_graph(graph: ComputationalGraph, input: [[u32; 2]; 2]) -> [[u32; 2]; 2] {
    // Define weights for the linear model (would come from model in real implementation)
    // Simple 2x1 weight matrix
    let weights = [[3, 0], [2, 0]];

    // Initialize result
    let mut result = 0;

    // Process each node in the graph
    for node in &graph.nodes {
        match node.op_type {
            OperationType::Input => {
                // Input node, nothing to do
            }
            OperationType::MatMul => {
                // For MatMul, execute matrix multiplication
                result = execute_matmul(input, weights);
                return uint2mat(result); // Early return with the result
            }
            _ => {
                // For any other operation, do nothing for now
            }
        }
    }

    // Default return if no operations were executed
    [[0, 0], [0, 0]]
}

fn mat2uint32(a: [[u32; 2]; 2]) -> u32 {
    // Convert a 2x2 matrix to a single u32 value by using 8 bits per element
    ((a[0][0] as u32 & 0xFF) << 24)
        | ((a[0][1] as u32 & 0xFF) << 16)
        | ((a[1][0] as u32 & 0xFF) << 8)
        | (a[1][1] as u32 & 0xFF)
}

fn uint2mat(a: u32) -> [[u32; 2]; 2] {
    // Convert a single u32 value back to a 2x2 matrix (8 bits per element)
    [
        [((a >> 24) & 0xFF) as u32, ((a >> 16) & 0xFF) as u32],
        [((a >> 8) & 0xFF) as u32, (a & 0xFF) as u32],
    ]
}

// Execute matrix multiplication operation
// This implements a 1x2 * 2x1 matrix multiplication (dot product)
fn execute_matmul(a: [[u32; 2]; 2], b: [[u32; 2]; 2]) -> u32 {
    // Pack the 2x2 matrix a into a 64-bit value
    // Each element is truncated to 16 bits
    let a_packed = mat2uint32(a);
    let b_packed = mat2uint32(b);

    use core::arch::asm;
    // Use ASM to perform addition
    unsafe {
        let mut result_matmul: u32 = 0;
        asm!(
            "REMU {val}, {rs1}, {rs2}",
            val = out(reg) result_matmul,
            rs1 = in(reg) a_packed,
            rs2 = in(reg) b_packed,
        );

        result_matmul
    }
}
