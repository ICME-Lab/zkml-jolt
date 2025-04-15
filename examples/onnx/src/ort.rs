use ndarray::Array;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file("examples/onnx/data/linear.onnx")?;

    let input: Array<f32, _> = Array::from_shape_vec((1, 1), vec![0.3f32])?;
    // ndarray を ONNX Runtime が要求する Tensor に変換
    let outputs = session.run(inputs![input]?)?;
    let output = &outputs[0].try_extract_tensor::<f32>()?;
    println!("Output: {:?}", output.as_slice().unwrap()[0]);

    Ok(())
}
