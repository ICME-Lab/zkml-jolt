#![cfg_attr(feature = "guest", no_std)]
#![allow(unused_assignments, asm_sub_register)]

// Indices for features
const FEATURE1_INDEX: usize = 0;
const FEATURE2_INDEX: usize = 1;

// Threshold constants (u8 version; e.g., 50 corresponds to original 0.5)
const T1: u8 = 50; // Threshold for splitting feature1
const T2: u8 = 30; // Next threshold when feature1 < T1
const T3: u8 = 70; // Next threshold when feature1 >= T1

// Leaf output values (u8 version)
// Note: original V1 was -1.0; since u8 cannot be negative, we use 10 as an example replacement.
const V1: u8 = 10; // Output value 1
const V2: u8 = 20; // Output value 2
const V3: u8 = 30; // Output value 3
const V4: u8 = 40; // Output value 4

// The jolt::provable attribute expects the input type to implement serde::Deserialize,
// so we use Vec<Vec<u8>> for a collection of feature vectors.
#[jolt::provable]
fn predict(data: [u8; 2]) -> u8 {
    predict_feature(&data)
}

// Predict for a single feature vector
fn predict_feature(features: &[u8]) -> u8 {
    if features[FEATURE1_INDEX] < T1 {
        // Left path
        if features[FEATURE2_INDEX] < T2 {
            V1 // Left-left leaf
        } else {
            V2 // Left-right leaf
        }
    } else {
        // Right path
        if features[FEATURE2_INDEX] < T3 {
            V3 // Right-left leaf
        } else {
            V4 // Right-right leaf
        }
    }
}

// The jolt::provable attribute expects the input type to implement serde::Deserialize,
// so we use Vec<Vec<u8>> for a collection of feature vectors.
#[jolt::provable]
fn predict_with_gbdt(data: [u8; 2]) -> u8 {
    predict_feature_with_gbdt(&data)
}

// GBDT
// call REM instead, because we don't have a GBDT instruction in the curren compiler toolchain
// Predict for a single feature vector using the GBDT instruction
fn predict_feature_with_gbdt(features: &[u8]) -> u8 {
    use core::arch::asm;
    let feature1 = features[FEATURE1_INDEX] as u32;
    let feature2 = features[FEATURE2_INDEX] as u32;

    unsafe {
        let mut val_gbdt: u32 = 0;
        asm!(
            "REM {val}, {rs1}, {rs2}",
            val = out(reg) val_gbdt,
            rs1 = in(reg) feature1,
            rs2 = in(reg) feature2,
        );

        val_gbdt as u8
    }
}
