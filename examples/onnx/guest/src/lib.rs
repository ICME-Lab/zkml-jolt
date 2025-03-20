use ndarray::Array2;
use anyhow::Context;
use tract_onnx::prelude::*;

#[jolt::provable]
fn onnx(model_path: &str, input_data: &[f32]) -> f32 {
    let input = input_to_array(input_data);
    let mut model = tract_onnx::onnx()
        .model_for_path(model_path)
        .context("Failed to load model").unwrap();

    // Explicitly specify input shape (batch size 1 with 5 features)
    model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 5)))
        .context("Failed to set input shape").unwrap();

    // Limit model output to only the first output (avoiding ZipMap)
    if model.output_outlets().unwrap().len() > 1 {
        println!("Multiple outputs detected. Using only the first output.");
        let first_output = model.output_outlets().unwrap()[0];
        model.set_output_outlets(&[first_output]).unwrap();
    }

    // Optimize model and make it runnable
    let model = model
        .into_optimized()
        .context("Failed to optimize model")
        .unwrap()
        .into_runnable()
        .context("Failed to convert to runnable model")
        .unwrap();
    let result = model.run(tvec!(input.clone())).unwrap();
    let tensor = &result[0];
    let tensor_data = tensor.to_array_view::<f32>().unwrap();
    tensor_data[[0, 0]]
}

fn input_to_array(input_data: &[f32]) -> TValue {
    let inputa_data_array = Array2::from_shape_vec((3, 5), input_data.to_vec()).unwrap();

    tract_ndarray::Array::from_shape_vec(
        (1, 5),
        inputa_data_array.iter().cloned().collect()
    )
    .context("Failed to reshape input data")
    .unwrap()
    .into_tvalue()
}
