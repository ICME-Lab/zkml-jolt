use anyhow::{Context, Result};
use tract_onnx::prelude::*;

pub fn main() -> Result<()> {
    println!("Linear ONNX Inference Demo");

    // Load the model
    let model_path = "examples/onnx/data/linear.onnx";
    println!("Loading model: {}", model_path);

    // Load the model and specify the correct input shape for the linear model
    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("Failed to load model")?;

    // Set input shape (batch size 1, input dimension 1)
    model
        .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1)))
        .context("Failed to set input shape")?;

    // Optimize model and make it runnable
    let model = model
        .into_optimized()
        .context("Failed to optimize model")?
        .into_runnable()
        .context("Failed to convert to runnable model")?;

    println!("Model ready for inference");

    // Simple test data
    let test_value: f32 = 3.0;

    // Create input tensor
    let input = tract_ndarray::arr1(&[test_value])
        .into_shape_with_order((1, 1))
        .unwrap()
        .into_tvalue();

    // Run inference
    let result = model.run(tvec!(input)).context("Failed to run inference")?;

    // Display results
    if let Some(tensor) = result.get(0) {
        if let Ok(view) = tensor.to_array_view::<f32>() {
            println!("Output: {:?}", view);
        } else {
            println!("Failed to access output as f32");
        }
    }

    println!("\nInference completed");
    Ok(())
}
