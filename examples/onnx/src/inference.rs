use anyhow::{Context, Result};
use ndarray::Array4;
use tract_onnx::prelude::*;

pub fn main() -> Result<()> {
    println!("CNN ONNX Inference Demo (Quantized Model)");

    // Load the quantized ONNX model
    let model_path = "examples/onnx/data/int8_cnn.onnx";
    println!("Loading model: {}", model_path);

    // Load the model and explicitly specify input shape
    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("Failed to load model")?;

    // Specify input shape for CNN (batch size 1, 1 channel, height 28, width 28)
    model
        .set_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, 28, 28)),
        )
        .context("Failed to set input shape")?;

    // Limit model output to only the first output
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
    println!("Model info: Input shape set to [1, 1, 28, 28]");

    // Create test data (integer input)
    let test_data = create_test_data();
    println!("Test data created: 3 samples of [1, 28, 28] images");

    // Set quantization parameters
    let input_scale = 1.0 / 255.0;
    let input_zero_point = 0;

    // Run inference for each sample
    for (i, sample) in test_data.iter().enumerate() {
        println!("\nProcessing sample {}", i + 1);

        // Convert integer input to floating point (dequantize)
        let mut float_input = Array4::<f32>::zeros((1, 1, 28, 28));
        for ((_batch, _channel, row, col), val) in float_input.indexed_iter_mut() {
            let idx = row * 28 + col;
            let int_val = sample[idx];
            *val = (int_val as f32 - input_zero_point as f32) * input_scale;
        }

        // Prepare input data
        let input = float_input.into_tvalue();

        // Run inference
        let result = model.run(tvec!(input)).context("Failed to run inference")?;

        // Parse results
        println!("  Number of outputs = {}", result.len());

        // Display results
        for (j, tensor) in result.iter().enumerate() {
            println!("  Output {}: Shape = {:?}", j, tensor.shape());

            // Try various types
            if let Ok(view) = tensor.to_array_view::<f32>() {
                // For floating point output (most likely)
                println!("    Output tensor (f32):");

                // For classification tasks, interpret as class scores
                if view.shape().len() == 2 && view.shape()[1] > 1 {
                    let mut max_idx = 0;
                    let mut max_val = view[[0, 0]];

                    // Find the index of maximum value (= predicted class)
                    for c in 0..view.shape()[1] {
                        if view[[0, c]] > max_val {
                            max_val = view[[0, c]];
                            max_idx = c;
                        }
                    }

                    println!(
                        "    Predicted class: {} (confidence: {:.6})",
                        max_idx, max_val
                    );

                    // Display all class scores
                    println!("    All class scores: ");
                    for c in 0..view.shape()[1] {
                        println!("      Class {}: {:.6}", c, view[[0, c]]);
                    }
                } else {
                    // For other shapes, display as is
                    println!("    Value = {:?}", view);
                }
            } else if let Ok(view) = tensor.to_array_view::<i64>() {
                println!("    Value (i64) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<i32>() {
                println!("    Value (i32) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<i8>() {
                println!("    Value (i8) = {:?}", view);
            } else if let Ok(view) = tensor.to_array_view::<u8>() {
                println!("    Value (u8) = {:?}", view);
            } else {
                println!("    Failed to convert value type");
                println!("    Data type: {:?}", tensor.datum_type());
            }
        }
    }

    println!("\nInference completed");
    Ok(())
}

// Function to create integer test data
fn create_test_data() -> Vec<Vec<u8>> {
    // Create 3 samples, each is a 28x28 image
    // For simplicity, generate random values (0-255)
    let mut rng = rand::thread_rng();

    (0..3)
        .map(|_| {
            (0..28 * 28)
                .map(|_| rand::Rng::gen_range(&mut rng, 0..=255) as u8)
                .collect()
        })
        .collect()
}

// Alternative function for when using actual test images
#[allow(dead_code)]
fn create_structured_test_data() -> Vec<Vec<u8>> {
    // Create more structured test data (e.g., simple shapes)

    // Image size
    const SIZE: usize = 28;

    // Background and foreground colors
    const BG: u8 = 0; // Background color (black)
    const FG: u8 = 255; // Foreground color (white)

    // Sample 1: Cross in the center
    let mut sample1 = vec![BG; SIZE * SIZE];
    for i in 0..SIZE {
        // Horizontal line
        sample1[i + (SIZE / 2) * SIZE] = FG;
        // Vertical line
        sample1[(SIZE / 2) + i * SIZE] = FG;
    }

    // Sample 2: Circle in the center
    let mut sample2 = vec![BG; SIZE * SIZE];
    let center = SIZE / 2;
    let radius = SIZE / 4;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = (x as isize - center as isize).abs() as f32;
            let dy = (y as isize - center as isize).abs() as f32;
            let distance = (dx * dx + dy * dy).sqrt();

            if distance > radius as f32 - 0.5 && distance < radius as f32 + 0.5 {
                sample2[x + y * SIZE] = FG;
            }
        }
    }

    // Sample 3: Diagonal lines
    let mut sample3 = vec![BG; SIZE * SIZE];
    for i in 0..SIZE {
        // Top-left to bottom-right
        sample3[i + i * SIZE] = FG;
        // Top-right to bottom-left
        sample3[(SIZE - 1 - i) + i * SIZE] = FG;
    }

    vec![sample1, sample2, sample3]
}
