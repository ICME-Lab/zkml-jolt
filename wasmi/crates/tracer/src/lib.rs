use crate::{args::Args, display::DisplayFuncType};
use anyhow::bail;
use common::rv_trace::{ELFInstruction, RVTraceRow};
use context::Context;
use std::process;
use utils::{
    get_invoked_func,
    print_execution_start,
    print_pretty_results,
    print_remaining_fuel,
    typecheck_args,
};
use wasmi::{Config, InstructionPtr};

pub mod args;
pub mod context;
pub mod display;
pub mod error;
pub mod utils;

#[cfg(test)]
mod tests;

pub fn trace(args: Args) -> anyhow::Result<Vec<RVTraceRow>> {
    // Build up WASM context and get the invoked function and validate i/o.
    let mut ctx = Context::new(&args)?;
    let (func_name, func) = get_invoked_func(&args, &ctx)?;
    let ty = func.ty(ctx.store());
    let func_args = utils::decode_func_args(&ty, args.func_args())?;
    let mut func_results = utils::prepare_func_results(&ty);
    typecheck_args(&func_name, &ty, &func_args)?;
    if args.verbose() {
        print_execution_start(args.wasm_file(), &func_name, &func_args);
    }
    if args.invoked().is_some() && ty.params().len() != args.func_args().len() {
        bail!(
            "invalid amount of arguments given to function {}. expected {} but received {}",
            DisplayFuncType::new(&func_name, &ty),
            ty.params().len(),
            args.func_args().len()
        )
    }

    // Execute the function with the given arguments.
    // The results are written to the mutable `func_results`.
    match func.call(ctx.store_mut(), &func_args, &mut func_results) {
        Ok(()) => {
            print_remaining_fuel(&args, &ctx);
            print_pretty_results(&func_results);
        }
        Err(error) => {
            if let Some(exit_code) = error.i32_exit_status() {
                // We received an exit code from the WASI program,
                // therefore we exit with the same exit code after
                // pretty printing the results.
                print_remaining_fuel(&args, &ctx);
                print_pretty_results(&func_results);
                process::exit(exit_code)
            }
            bail!("failed during execution of {func_name}: {error}")
        }
    };

    // Extract the execution trace from the context.
    let mut rows = ctx.store().tracer.rows.try_borrow_mut().unwrap();
    let mut output = Vec::new();
    output.append(&mut rows);
    drop(rows);

    Ok(output)
}

/// Gets the code_map from the WASM module & converts it to `Vec<ELFInstruction>`.
///
/// # Returns
///
/// - `Vec<ELFInstruction>`: The decoded WASM instructions extracted from the WASM bytecode.
/// - `Vec<(u64, u8)>`: The memory map of the actual WASM module i.e. the actual memory bytes of the WASM module.
#[tracing::instrument(skip_all)]
pub fn decode(wasm_bytecode: &[u8]) -> (Vec<ELFInstruction>, Vec<(u64, u8)>) {
    // Initiate the [`EngineFunc`] with the given bytecode.
    // This is a workaround to get the code_map from the WASM module.
    let engine = wasmi::Engine::new(&Config::default());
    let _module = wasmi::Module::new(&engine, wasm_bytecode).unwrap();

    // Get the `&[Instructions]` using the intialized [`EngineFunc`].
    let instructions = engine.instructions();

    // Keep track of the pc/instruction pointer.
    let mut pc = InstructionPtr::new(instructions.as_ptr());
    let base_addr = InstructionPtr::new(instructions.as_ptr());
    const SKIP: usize = 1;

    // Convert the instructions to [`Vec<ELFInstruction>`].
    const ACCOUNT_FOR_NOOP: u64 = 1;
    let elf_instructions: Vec<ELFInstruction> = instructions
        .iter()
        .copied()
        .map(|instr| {
            // We calculate the instruction address by taking the offset from the base address (first instruction).
            //
            // # Note
            //
            // We add `ACCOUNT_FOR_NOOP` to the instruction address to account for the fact that we prepend a NOOP instruction to the bytecode.
            let instruction_address = pc.offset_from(base_addr) as u64 + ACCOUNT_FOR_NOOP;
            debug_assert_eq!(instr, *pc.get());
            let elf_instruction = instr.trace(instruction_address);
            pc.add(SKIP);
            elf_instruction
        })
        .collect();
    // TODO: INIT_MEMORY?!!!
    (elf_instructions, vec![])
}

#[cfg(test)]
pub mod test_lib {
    use std::fs;

    use crate::{
        tests::{add_sub_mul_32_wasm_program, bitwise_arith_wasm_program},
        trace,
    };

    #[test]
    fn test_add_sub_mul_32() {
        let execution_trace = trace(add_sub_mul_32_wasm_program()).unwrap();
        println!("Execution Trace: {execution_trace:#?}");
    }

    #[test]
    fn test_bitwise_arith() {
        let execution_trace = trace(bitwise_arith_wasm_program()).unwrap();
        println!("Execution Trace: {execution_trace:#?}");
    }

    #[test]
    fn print_code_map() {
        let wasm_bytecode = fs::read("../../../wasms/bitwise_arith.wat").unwrap();
        let engine = wasmi::Engine::new(&wasmi::Config::default());
        let _module = wasmi::Module::new(&engine, wasm_bytecode).unwrap();
        let instructions = engine.instructions();
        println!("Instructions: {instructions:#?}");
    }

    // #[test]
    // fn test_reward() {
    //     let stake = "1500".to_string(); // Amount of LP tokens or liquidity staked by the user.
    //     let duration_boost = "3".to_string(); // Boost multiplier based on how long the stake was held (e.g., 3 = 3 months).
    //     let volume_boost = "2".to_string(); // Additional multiplier based on trading volume in the pool during the staking period.
    //     let penalty = "500".to_string(); // Penalty applied for early withdrawal or performance issues (e.g., protocol downgrade).

    //     let file_path = "binaries/calculate-reward.wasm";
    //     let args = Args::new(
    //         file_path,
    //         "main",
    //         vec![stake, duration_boost, volume_boost, penalty],
    //     );
    //     run(args).unwrap();
    // }

    // #[test]
    // fn test_energy_consumption() {
    //     let total_produced = "5000".to_string(); // Total energy produced by the microgrid in some time frame (e.g., in watt-hours).
    //     let total_consumed = "4900".to_string(); // Total energy consumed by all devices in the microgrid for the same period.
    //     let device_count = "100".to_string(); // Number of IoT devices or meters in the network.
    //     let baseline_price = "100".to_string(); // A baseline price or factor used for further calculations (e.g., cost per watt-hour or an
    //                                             // index).
    //     let file_path = "binaries/energy_usage_32.wasm";
    //     let args = Args::new(
    //         file_path,
    //         "main",
    //         vec![total_produced, total_consumed, device_count, baseline_price],
    //     );
    //     run(args).unwrap();
    // }
}
