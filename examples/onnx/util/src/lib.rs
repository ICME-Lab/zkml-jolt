extern crate alloc;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Represents an operation in the computational graph
/// 0 = Input
/// 1 = Constant
/// 2 = MatMul
/// 3 = Add
/// 4 = Relu
/// 5 = Reshape
/// 6 = TreeEnsembleClassifier
/// 7+ = Unknown
pub type OperationType = usize;

// Constants for operation types
pub const OP_INPUT: usize = 0;
pub const OP_CONSTANT: usize = 1;
pub const OP_MATMUL: usize = 2;
pub const OP_ADD: usize = 3;
pub const OP_RELU: usize = 4;
pub const OP_RESHAPE: usize = 5;
pub const OP_TREE_ENSEMBLE_CLASSIFIER: usize = 6;
pub const OP_UNKNOWN: usize = 7;

/// Represents a node in the computational graph
#[derive(Serialize, Deserialize)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: OperationType,
    pub name: String,
}

/// Represents the computational graph
#[derive(Serialize, Deserialize)]
pub struct ComputationalGraph {
    pub nodes: Vec<GraphNode>,
    pub input_count: usize,
    pub output_count: usize,
}

#[cfg(feature = "host")]
use anyhow::Result;
#[cfg(feature = "host")]
use std::collections::HashMap;
#[cfg(feature = "host")]
use std::path::Path;
#[cfg(feature = "host")]
use tract_onnx::prelude::*;

#[cfg(feature = "host")]
impl ComputationalGraph {
    /// Creates a new empty computational graph
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: Vec::new(),
            input_count: 0,
            output_count: 0,
        }
    }

    /// Print the computational graph
    pub fn print(&self) {
        println!("Computational Graph:");
        println!("Number of inputs: {}", self.input_count);
        println!("Number of outputs: {}", self.output_count);
        println!("Nodes ({}): ", self.nodes.len());

        // Count operation types
        let mut op_counts: HashMap<OperationType, usize> = HashMap::new();
        for node in &self.nodes {
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        }

        // Print node count by type
        println!("\nOperation types:");
        for (op_type, count) in op_counts {
            println!("  {:?}: {} nodes", op_type, count);
        }

        // Print node details
        println!("\nNode details:");
        for node in &self.nodes {
            println!("  Node {}: {}", node.id, node.name);
            println!("    Type: {:?}", node.op_type);
        }
    }
}

/// ONNX Parser
#[cfg(feature = "host")]
pub struct ONNXParser;

#[cfg(feature = "host")]
impl ONNXParser {
    /// Load an ONNX model from a file
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ComputationalGraph> {
        // Create an empty computational graph
        let mut graph = ComputationalGraph::new();

        // Load the ONNX model using tract
        let model = tract_onnx::onnx().model_for_path(path)?;

        // Count inputs and outputs
        graph.input_count = model.inputs.len();
        graph.output_count = model.outputs.len();

        // Process each node in the model
        for (id, node) in model.nodes.iter().enumerate() {
            // Get the operation name
            let op_name = node.op.name();

            // Map operation name to OperationType
            let op_type = match op_name.as_ref() {
                "Source" => OP_INPUT,
                "Const" => OP_CONSTANT,
                "MatMul" => OP_MATMUL,
                "Add" => OP_ADD,
                "Relu" => OP_RELU,
                "Reshape" => OP_RESHAPE,
                "TreeEnsembleClassifier" => OP_TREE_ENSEMBLE_CLASSIFIER,
                _ => OP_UNKNOWN,
            };

            // Create the graph node
            let graph_node = GraphNode {
                id,
                op_type,
                name: node.name.clone(),
            };

            // Add the node to the graph
            graph.nodes.push(graph_node);
        }

        Ok(graph)
    }
}
