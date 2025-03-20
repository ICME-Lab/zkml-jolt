use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use tract_onnx::prelude::*;

/// Represents an operation in the computational graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperationType {
    Input,
    Constant,
    MatMul,
    Add,
    Relu,
    Unknown(String),
}

/// Represents a node in the computational graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: OperationType,
    pub name: String,
}

/// Represents the computational graph
#[derive(Debug, Clone)]
pub struct ComputationalGraph {
    pub nodes: Vec<GraphNode>,
    pub input_count: usize,
    pub output_count: usize,
}

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
pub struct ONNXParser;

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
                "Source" => OperationType::Input,
                "Const" => OperationType::Constant,
                "MatMul" => OperationType::MatMul,
                "Add" => OperationType::Add,
                "Relu" => OperationType::Relu,
                _ => OperationType::Unknown(op_name.to_string()),
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

pub fn main() {
    println!("ONNX Parser Implementation Demo");

    // Create a full path for the ONNX model
    let model_path = "examples/onnx/data/cnn.onnx";

    // Parse the ONNX model using our custom parser
    println!("\nParsing ONNX model with our custom parser...");
    match ONNXParser::load_model(model_path) {
        Ok(graph) => {
            println!("\n=== ONNX Model Parsed Successfully ===");
            println!("Model has {} nodes", graph.nodes.len());
            println!("Model has {} input(s)", graph.input_count);
            println!("Model has {} output(s)", graph.output_count);

            // Print detailed information about the computational graph
            println!("\n=== Computational Graph Details ===");
            graph.print();

            println!("\nONNX parsing completed successfully!");
        }
        Err(e) => {
            println!("Failed to parse ONNX model: {}", e);
        }
    }

    // Now run tract's native parser to compare
    println!("\nParsing with tract's native parser for comparison...");
    match tract_onnx::onnx().model_for_path(model_path) {
        Ok(model) => {
            println!("Tract parsed the model successfully.");
            println!("Model has {} inputs", model.inputs.len());
            println!("Model has {} outputs", model.outputs.len());
            println!("Model has {} nodes", model.nodes.len());
        }
        Err(e) => {
            println!("Tract failed to parse the model: {}", e);
        }
    }
}
