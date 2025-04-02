#![allow(unused_assignments, asm_sub_register)]

use onnx_util::{ComputationalGraph, OperationType};

#[jolt::provable]
fn execute_graph(graph: ComputationalGraph, input: [[u32; 2]; 2]) -> u32 {
    // Define weights for the linear model (would come from model in real implementation)
    // Simple 2x1 weight matrix
    let weights = [[3], [2]];

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
                return result; // Early return with the result
            }
            _ => {
                // For any other operation, do nothing for now
            }
        }
    }

    // Default return if no operations were executed
    result
}

// Execute matrix multiplication operation
// This implements a 1x2 * 2x1 matrix multiplication (dot product)
fn execute_matmul(a: [[u32; 2]; 2], b: [[u32; 1]; 2]) -> u32 {
    // For our linear model, we're doing a dot product: a[0][0]*b[0][0] + a[0][1]*b[1][0]
    let mut result = 0;

    // Calculate mul
    let mul1 = a[0][0] * b[0][0];
    let mul2 = a[0][1] * b[1][0];
    // Sum the products
    result = execute_add_with_asm(mul1, mul2);

    result
}

// Execute addition operation using ASM instructions
fn execute_add_with_asm(a: u32, b: u32) -> u32 {
    use core::arch::asm;
    // Use ASM to perform addition
    unsafe {
        let mut result_add: u32 = 0;
        asm!(
            "ADD {val}, {rs1}, {rs2}",
            val = out(reg) result_add,
            rs1 = in(reg) a,
            rs2 = in(reg) b,
        );

        result_add
    }
}
