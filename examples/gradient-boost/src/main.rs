use rand::random;
use std::time::Instant;

pub fn main() {
    for _ in 0..10 {
        println!("===========");
        let test_data = [random::<u8>(), random::<u8>()];
        println!("Test data: {:?}", test_data);
        // Run standard implementation
        let output_standard = run_standard_implementation(&test_data);

        // Run GBDT implementation
        let output_custom = run_gbdt_infer_implementation(&test_data);

        assert_eq!(output_standard, output_custom);

        println!("Output: {}", output_standard);
        println!("===========");
        println!("");
    }
}

fn run_standard_implementation(test_data: &[u8; 2]) -> u8 {
    println!("\n=== Standard Implementation ===");

    let (prove_predict, verify_predict) = guest::build_predict();
    // let program_summary = guest::analyze_predict(test_data.clone());
    // program_summary
    //     .write_to_file("gbm_standard.txt".into())
    //     .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_predict(test_data.clone());
    let prover_time = now.elapsed().as_secs_f64();
    println!("Prover runtime: {} s", prover_time);

    let is_valid = verify_predict(proof);
    assert!(is_valid, "Invalid output for standard implementation");
    output
}

fn run_gbdt_infer_implementation(test_data: &[u8; 2]) -> u8 {
    println!("\n=== GBDT Implementation ===");

    let (prove_predict, verify_predict) = guest::build_predict_with_gbdt();
    // let program_summary = guest::analyze_predict_with_gbdt(test_data.clone());
    // program_summary
    //     .write_to_file("gbm_gbdt_infer.txt".into())
    //     .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_predict(test_data.clone());
    let prover_time = now.elapsed().as_secs_f64();
    println!("Prover runtime: {} s", prover_time);

    let is_valid = verify_predict(proof);
    assert!(is_valid, "Invalid output for GBDT implementation");
    output
}
