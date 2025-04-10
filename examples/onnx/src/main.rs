use onnx_util::ONNXParser;

pub fn main() {
    let input_data = [[50, 20], [30, 40]];
    // Use cnn.onnx instead of xgboost.onnx as per user's instruction
    let model_path = "examples/onnx/data/linear.onnx";

    // Parse the ONNX model to get the computational graph
    println!("Parsing ONNX model to get computational graph...");
    let graph = match ONNXParser::load_model(model_path) {
        Ok(g) => {
            println!("Successfully parsed the model.");
            g
        }
        Err(e) => {
            println!("Failed to parse ONNX model: {}", e);
            return;
        }
    };

    // Execute the guest code with the computational graph
    let (prove_execute_graph, verify_execute_graph) = guest::build_execute_graph();
    let (output, proof) = prove_execute_graph(graph, input_data);
    let is_valid = verify_execute_graph(proof);
    assert!(is_valid, "Invalid output for execute_graph");
    println!("Output from execute_graph: {:?}", output);
}
