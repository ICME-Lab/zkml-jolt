use super::*;
use std::borrow::Borrow;
use wasmi::{core::ValType, FuncType};

pub fn add_sub_mul_32_wasm_program() -> Args {
    let file_path = "./wasms/add_sub_mul_32.wat";
    Args::new(file_path, "main", testing_func_args())
}

pub fn bitwise_arith_wasm_program() -> Args {
    let file_path = "./wasms/bitwise_arith.wat";
    Args::new(file_path, "main", testing_func_args())
}

pub fn shifts_arith_wasm_program() -> Args {
    let file_path = "./wasms/shifts_arith.wat";
    Args::new(file_path, "main", testing_func_args())
}

pub fn lt_wasm_program() -> Args {
    let file_path = "./wasms/lt.wat";
    Args::new(file_path, "main", testing_func_args())
}

fn testing_func_args() -> Vec<String> {
    let stake = "1500".to_string(); // Amount of LP tokens or liquidity staked by the user.
    let duration_boost = "3".to_string(); // Boost multiplier based on how long the stake was held (e.g., 3 = 3 months).
    let volume_boost = "2".to_string(); // Additional multiplier based on trading volume in the pool during the staking period.
    let penalty = "500".to_string(); // Penalty applied for early withdrawal or performance issues (e.g., protocol downgrade).
    vec![stake, duration_boost, volume_boost, penalty]
}

fn assert_display(func_type: impl Borrow<FuncType>, expected: &str) {
    assert_eq!(
        format!("{}", DisplayFuncType::from(func_type.borrow())),
        String::from(expected),
    );
}

macro_rules! func_ty {
    ($params:expr, $results:expr $(,)?) => {{
        FuncType::new($params, $results)
    }};
}

#[test]
fn display_0in_0out() {
    assert_display(func_ty!([], []), "fn()");
}

#[test]
fn display_1in_0out() {
    assert_display(func_ty!([ValType::I32], []), "fn(i32)");
}

#[test]
fn display_0in_1out() {
    assert_display(func_ty!([], [ValType::I32]), "fn() -> i32");
}

#[test]
fn display_1in_1out() {
    assert_display(func_ty!([ValType::I32], [ValType::I32]), "fn(i32) -> i32");
}

#[test]
fn display_4in_0out() {
    assert_display(
        func_ty!([ValType::I32, ValType::I64, ValType::F32, ValType::F64], []),
        "fn(i32, i64, f32, f64)",
    );
}

#[test]
fn display_0in_4out() {
    assert_display(
        func_ty!([], [ValType::I32, ValType::I64, ValType::F32, ValType::F64]),
        "fn() -> (i32, i64, f32, f64)",
    );
}

#[test]
fn display_4in_4out() {
    assert_display(
        func_ty!(
            [ValType::I32, ValType::I64, ValType::F32, ValType::F64],
            [ValType::I32, ValType::I64, ValType::F32, ValType::F64],
        ),
        "fn(i32, i64, f32, f64) -> (i32, i64, f32, f64)",
    );
}
