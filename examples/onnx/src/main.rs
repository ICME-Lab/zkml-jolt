use anyhow::{Context, Result};
use ndarray::Array2;
use tract_onnx::prelude::*;

pub fn main() {
    let input_data = vec![
        0.5, 0.2, 0.1, 0.8, 0.3,  // Sample 1
        0.1, 0.7, 0.5, 0.2, 0.9,  // Sample 2
        0.3, 0.4, 0.6, 0.7, 0.2,  // Sample 3
    ];
    let model_path = "examples/onnx/data/xgboost.onnx";

    let (prove_onnx, verify_onnx) = guest::build_onnx();
    let (output, proof) = prove_onnx(model_path, &input_data);
    let is_valid = verify_onnx(proof);
    assert!(is_valid, "Invalid output for ONNX implementation");
    println!("Output: {}", output);
}

#[allow(dead_code)]
fn load_and_inference() -> Result<()> {
    println!("XGBoost ONNX Inference Demo");

    // Load ONNX model
    let model_path = "examples/onnx/data/xgboost.onnx";
    println!("Loading model: {}", model_path);

    // Load the model and explicitly specify input shape
    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("Failed to load model")?;

    // Explicitly specify input shape (batch size 1 with 5 features)
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 5)))
        .context("Failed to set input shape")?;

    // Limit model output to only the first output (avoiding ZipMap)
    if model.output_outlets()?.len() > 1 {
        println!("Multiple outputs detected. Using only the first output.");
        let first_output = model.output_outlets()?[0];
        model.set_output_outlets(&[first_output])?;
    }

    // Optimize model and make it runnable
    let model = model
        .into_optimized()
        .context("Failed to optimize model")?
        .into_runnable()
        .context("Failed to convert to runnable model")?;

    // Display model information
    println!("Model info: Input shape set");

    // Create test data (samples with 5 features)
    let test_data = create_test_data();
    println!("Test data: {:?}", test_data);

    // Run inference
    for (i, sample) in test_data.outer_iter().enumerate() {
        // Prepare input data
        let input = tract_ndarray::Array::from_shape_vec(
            (1, 5),
            sample.iter().cloned().collect(),
        )
        .context("Failed to reshape input data")?
        .into_tvalue();

        // Run inference
        let result = model.run(tvec!(input))
            .context("Failed to run inference")?;

        // Parse results
        println!("Sample {}: Number of results = {}", i, result.len());

        // Display results - try various types
        for (j, tensor) in result.iter().enumerate() {
            println!("  Output {}: Shape = {:?}", j, tensor.shape());

            // Try various types
            if let Ok(view) = tensor.to_array_view::<i64>() {
                println!("    Value (i64) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<f32>() {
                println!("    Value (f32) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<i32>() {
                println!("    Value (i32) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<u8>() {
                println!("    Value (u8) = {:?}", view);
            } else {
                println!("    Failed to convert value type");
                println!("    Data type: {:?}", tensor.datum_type());
            }
        }
    }

    println!("Inference completed");
    Ok(())
}

// Function to create test data
fn create_test_data() -> Array2<f32> {
    // Create 3 samples with 5 features each
    // Note: This is random test data
    Array2::from_shape_vec(
        (3, 5),
        vec![
            0.5, 0.2, 0.1, 0.8, 0.3,  // Sample 1
            0.1, 0.7, 0.5, 0.2, 0.9,  // Sample 2
            0.3, 0.4, 0.6, 0.7, 0.2,  // Sample 3
        ],
    )
    .unwrap()
}