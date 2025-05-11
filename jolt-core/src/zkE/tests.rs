use super::wasm_host::WASMProgram;

pub fn testing_wasm_program() -> WASMProgram {
    // WASM inputs
    let stake = "1500".to_string(); // Amount of LP tokens or liquidity staked by the user.
    let duration_boost = "3".to_string(); // Boost multiplier based on how long the stake was held (e.g., 3 = 3 months).
    let volume_boost = "2".to_string(); // Additional multiplier based on trading volume in the pool during the staking period.
    let penalty = "500".to_string(); // Penalty applied for early withdrawal or performance issues (e.g., protocol downgrade).
    let wasm_program = WASMProgram {
        func: "main".to_string(),
        inputs: vec![stake, duration_boost, volume_boost, penalty],
        file_path: "./wasms/add_sub_mul_32.wat".to_string(),
    };
}
