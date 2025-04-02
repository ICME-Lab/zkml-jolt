#![allow(unused_assignments, asm_sub_register)]

use onnx_util::ComputationalGraph;

#[jolt::provable]
fn execute_graph(graph: ComputationalGraph, input_data: [u32; 2]) -> u32 {
    // Process each node in the graph
    for node in &graph.nodes[1..] {
        let a = input_data[0]; // Placeholder for actual input data
        let b = input_data[1]; // Placeholder for actual input data

        let _ = node.op_type;

        // For now, we'll just execute an add operation for any non-input node
        return execute_add_with_asm(a, b);
    }

    // Default return if no operations were executed
    0
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
