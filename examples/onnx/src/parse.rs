use clap::Parser;
use onnx_util::ONNXParser;
use tract_onnx::prelude::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value_t = String::from("examples/onnx/data/network_reg.onnx"))]
    model_path: String,
}

pub fn main() {
    let args = Args::parse();
    let model_path = args.model_path.as_str();
    println!("ONNX Parser Implementation Demo");

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
