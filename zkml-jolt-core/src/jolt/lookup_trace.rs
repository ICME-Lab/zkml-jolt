use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};

use crate::jolt::instruction::{add::ADD, mul::MUL, sub::SUB};
use jolt_core::jolt::{
    instruction::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};

// pub const WORD_SIZE: usize = 32;

macro_rules! test_const_generic {
    ( $variant:ident : $inner:ident <WORD_SIZE>) => {};
}

test_const_generic!(Add: ADD<WORD_SIZE>);

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
     //   const $word_size:ident,
        trait $trait_name:ident,
        $($variant:ident : $inner:ident < WORD_SIZE >),+ $(,)?
    ) => {
        #[derive(Debug)]
        pub enum $enum_name<const WORD_SIZE: usize> {
            $(
                $variant($inner<WORD_SIZE>),
            )+
        }

        impl<const WORD_SIZE: usize> $trait_name<WORD_SIZE> for $enum_name<WORD_SIZE> {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum ONNXLookup,
    trait LookupQuery,
    Add: ADD<WORD_SIZE>,
    Sub: SUB<WORD_SIZE>,
    Mul: MUL<WORD_SIZE>,
);

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for ONNXLookup<WORD_SIZE> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ONNXLookup::Add(add) => add.lookup_table(),
            ONNXLookup::Sub(sub) => sub.lookup_table(),
            ONNXLookup::Mul(mul) => mul.lookup_table(),
        }
    }
}

pub trait LookupTrace<const WORD_SIZE: usize> {
    fn to_lookup(&self) -> Option<ONNXLookup<WORD_SIZE>>;
}

impl<const WORD_SIZE: usize> LookupTrace<WORD_SIZE> for ONNXCycle {
    fn to_lookup(&self) -> Option<ONNXLookup<WORD_SIZE>> {
        match self.instr.opcode {
            ONNXOpcode::Add => Some(ONNXLookup::Add(ADD(self.ts1_val(), self.ts2_val()))),
            ONNXOpcode::Sub => Some(ONNXLookup::Sub(SUB(self.ts1_val(), self.ts2_val()))),
            ONNXOpcode::Mul => Some(ONNXLookup::Mul(MUL(self.ts1_val(), self.ts2_val()))),
            _ => None,
        }
    }
}
