pub fn main() {
    let test_value = 0.4f32;
    let model_path = "examples/onnx/data/linear.onnx";

    let (prove_onnx, verify_onnx) = guest::build_onnx();
    let (output, proof) = prove_onnx(model_path, test_value);
    let is_valid = verify_onnx(proof);
    assert!(is_valid, "Invalid output for ONNX implementation");
    println!("Output: {}", output);
}
