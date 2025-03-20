pub fn main() {
    let input_data = vec![
        0.5, 0.2, 0.1, 0.8, 0.3, // Sample 1
        0.1, 0.7, 0.5, 0.2, 0.9, // Sample 2
        0.3, 0.4, 0.6, 0.7, 0.2, // Sample 3
    ];
    let model_path = "examples/onnx/data/xgboost.onnx";

    let (prove_onnx, verify_onnx) = guest::build_onnx();
    let (output, proof) = prove_onnx(model_path, &input_data);
    let is_valid = verify_onnx(proof);
    assert!(is_valid, "Invalid output for ONNX implementation");
    println!("Output: {}", output);
}
