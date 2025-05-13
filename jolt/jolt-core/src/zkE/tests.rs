use super::wasm_host::WASMProgram;

pub fn add_sub_mul_wasm_program() -> WASMProgram {
    WASMProgram {
        func: "main".to_string(),
        inputs: wasm_test_inputs(),
        file_path: "../wasms/add_sub_mul_32.wat".to_string(),
    }
}

pub fn bitwise_arith_wasm_program() -> WASMProgram {
    WASMProgram {
        func: "main".to_string(),
        inputs: wasm_test_inputs(),
        file_path: "../wasms/bitwise_arith.wat".to_string(),
    }
}

pub fn shifts_arith_wasm_program() -> WASMProgram {
    WASMProgram {
        func: "main".to_string(),
        inputs: wasm_test_inputs(),
        file_path: "../wasms/shifts_arith.wat".to_string(),
    }
}

// WASM inputs
fn wasm_test_inputs() -> Vec<String> {
    let stake = "1500".to_string(); // Amount of LP tokens or liquidity staked by the user.
    let duration_boost = "3".to_string(); // Boost multiplier based on how long the stake was held (e.g., 3 = 3 months).
    let volume_boost = "2".to_string(); // Additional multiplier based on trading volume in the pool during the staking period.
    let penalty = "500".to_string(); // Penalty applied for early withdrawal or performance issues (e.g., protocol downgrade).
    vec![stake, duration_boost, volume_boost, penalty]
}
