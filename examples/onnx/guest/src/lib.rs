#![allow(unused_assignments, asm_sub_register)]

use onnx_util::{ComputationalGraph, OP_INPUT};

#[jolt::provable]
fn execute_graph(input_data: [u32; 2]) -> u32 {
    // Process each node in the graph
    // for node in &graph.nodes {
    //     // Skip input nodes
    //     if node.op_type == OP_INPUT {
    //         continue;
    //     }

    //     let a = input_data[0]; // Placeholder for actual input data
    //     let b = input_data[1]; // Placeholder for actual input data

    //     // For now, we'll just execute an add operation for any non-input node
    //     return execute_add_with_asm(a, b);
    // }

    let a = input_data[0]; // Placeholder for actual input data
    let b = input_data[1]; // Placeholder for actual input data

    // For now, we'll just execute an add operation for any non-input node
    return execute_add_with_asm(a, b);

    // Default return if no operations were executed
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
