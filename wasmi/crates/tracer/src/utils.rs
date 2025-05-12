use std::path::Path;

use crate::{
    args::Args,
    context::Context,
    display::{
        DisplayExportedFuncs,
        DisplayFuncType,
        DisplaySequence,
        DisplayValue,
        DisplayValueType,
    },
};
use anyhow::{anyhow, bail, Error};
use wasmi::{
    core::{ValType, F32, F64, V128},
    Func,
    FuncType,
    Val,
};

/// Returns a [`Val`] buffer capable of holding the return values.
///
/// The returned buffer can be used as function results for [`Func::call`](`wasmi::Func::call`).
pub fn prepare_func_results(ty: &FuncType) -> Box<[Val]> {
    ty.results().iter().copied().map(Val::default).collect()
}

/// Decode the given `args` for the [`FuncType`] `ty`.
///
/// Returns the decoded `args` as a slice of [`Val`] which can be used
/// as function arguments for [`Func::call`][`wasmi::Func::call`].
///
/// # Errors
///
/// - If there is a type mismatch between `args` and the expected [`ValType`] by `ty`.
/// - If too many or too few `args` are given for [`FuncType`] `ty`.
/// - If unsupported [`ExternRef`] or [`FuncRef`] types are encountered.
///
/// [`FuncRef`]: wasmi::FuncRef
/// [`ExternRef`]: wasmi::ExternRef
pub fn decode_func_args(ty: &FuncType, args: &[String]) -> Result<Box<[Val]>, Error> {
    ty.params()
        .iter()
        .zip(args)
        .enumerate()
        .map(|(n, (param_type, arg))| {
            macro_rules! make_err {
                () => {
                    |_| {
                        anyhow!(
                            "failed to parse function argument \
                            {arg} at index {n} as {}",
                            DisplayValueType::from(param_type)
                        )
                    }
                };
            }
            match param_type {
                ValType::I32 => arg.parse::<i32>().map(Val::from).map_err(make_err!()),
                ValType::I64 => arg.parse::<i64>().map(Val::from).map_err(make_err!()),
                ValType::F32 => arg
                    .parse::<f32>()
                    .map(F32::from)
                    .map(Val::from)
                    .map_err(make_err!()),
                ValType::F64 => arg
                    .parse::<f64>()
                    .map(F64::from)
                    .map(Val::from)
                    .map_err(make_err!()),
                ValType::V128 => arg
                    .parse::<u128>()
                    .map(V128::from)
                    .map(Val::from)
                    .map_err(make_err!()),
                ValType::FuncRef => {
                    bail!("the wasmi CLI cannot take arguments of type funcref")
                }
                ValType::ExternRef => {
                    bail!("the wasmi CLI cannot take arguments of type externref")
                }
            }
        })
        .collect::<Result<Box<[_]>, _>>()
}

/// Prints the remaining fuel so far if fuel metering was enabled.
pub fn print_remaining_fuel(args: &Args, ctx: &Context) {
    if let Some(given_fuel) = args.fuel() {
        let remaining = ctx
            .store()
            .get_fuel()
            .unwrap_or_else(|error| panic!("could not get the remaining fuel: {error}"));
        let consumed = given_fuel.saturating_sub(remaining);
        println!("fuel consumed: {consumed}, fuel remaining: {remaining}");
    }
}

/// Performs minor typecheck on the function signature.
///
/// # Note
///
/// This is not strictly required but improve error reporting a bit.
///
/// # Errors
///
/// If too many or too few function arguments were given to the invoked function.
pub fn typecheck_args(
    func_name: &str,
    func_ty: &FuncType,
    args: &[Val],
) -> Result<(), anyhow::Error> {
    if func_ty.params().len() != args.len() {
        bail!(
            "invalid amount of arguments given to function {}. expected {} but received {}",
            DisplayFuncType::new(func_name, func_ty),
            func_ty.params().len(),
            args.len()
        )
    }
    Ok(())
}

/// Returns the invoked named function or the WASI entry point to the Wasm module if any.
///
/// # Errors
///
/// - If the function given via `--invoke` could not be found in the Wasm module.
/// - If `--invoke` was not given and no WASI entry points were exported.
pub fn get_invoked_func(args: &Args, ctx: &Context) -> Result<(String, Func), anyhow::Error> {
    match args.invoked() {
        Some(func_name) => {
            let func = ctx
                .get_func(func_name)
                .map_err(|error| anyhow!("{error}\n\n{}", DisplayExportedFuncs::from(ctx)))?;
            let func_name = func_name.into();
            Ok((func_name, func))
        }
        None => {
            // No `--invoke` flag was provided so we try to find
            // the conventional WASI entry points `""` and `"_start"`.
            if let Ok(func) = ctx.get_func("") {
                Ok(("".into(), func))
            } else if let Ok(func) = ctx.get_func("_start") {
                Ok(("_start".into(), func))
            } else {
                bail!(
                    "did not specify `--invoke` and could not find exported WASI entry point functions\n\n{}",
                    DisplayExportedFuncs::from(ctx)
                )
            }
        }
    }
}

/// Prints a signalling text that Wasm execution has started.
pub fn print_execution_start(wasm_file: &Path, func_name: &str, func_args: &[Val]) {
    println!(
        "executing File({wasm_file:?})::{func_name}({}) ...",
        DisplaySequence::new(", ", func_args.iter().map(DisplayValue::from))
    );
}

/// Prints the results of the Wasm computation in a human readable form.
pub fn print_pretty_results(results: &[Val]) {
    for result in results {
        println!("{}", DisplayValue::from(result))
    }
}
