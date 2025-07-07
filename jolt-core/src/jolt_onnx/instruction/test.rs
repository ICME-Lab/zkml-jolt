use std::collections::HashMap;

use ark_std::test_rng;
use common::constants::REGISTER_COUNT;
use rand_core::RngCore;

use crate::jolt_onnx::{common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator}, instruction::VirtualInstructionSequence, tracer::tensor::QuantizedTensor};


/// Tests the consistency and correctness of a virtual instruction sequence.
/// In detail:
/// 1. Sets the references to given values for `x` and `y`.
/// 2. Constructs an `ONNXTraceRow` with the provided opcode and reference values.
/// 3. Generates the virtual instruction sequence using the specified instruction type.
/// 4. Iterates over each row in the virtual sequence and validates the state changes.
/// 5. Verifies that the references `r_x` and `r_y` have not been modified (not clobbered).
/// 6. Ensures that the result of the instruction sequence is correctly written to the `rd` reference.
/// 7. Checks that no unintended modifications have been made to other references.
pub fn jolt_onnx_virtual_sequence_test<I: VirtualInstructionSequence>(opcode: Operator) {
    let mut rng = test_rng();

    for _ in 0..1000 {
        let r_x = rng.next_u64().to_string();
        let r_y = rng.next_u64().to_string();
        let mut rd = rng.next_u64().to_string();
        while rd == "0".to_string() {
            rd = rng.next_u64().to_string();
        }
        let x = if r_x == "0".to_string() { 0 } else { rng.next_u32() as i8 };
        let y = if r_y == r_x {
            x
        } else if r_y == "0".to_string() {
            0
        } else {
            rng.next_u32() as i8
        };
        let result = I::sequence_output(x, y);

        let mut registers : HashMap<String, QuantizedTensor> = HashMap::new();
        registers.insert(r_x.clone(), QuantizedTensor::from(x));
        registers.insert(r_y.clone(), QuantizedTensor::from(y));

        let trace_row = ONNXTraceRow {
            instruction: ONNXInstruction {
                opcode,
                input_refs: vec![r_x.clone(), r_y.clone()],
                output_refs: vec![rd.clone()],
                attributes: None,
            },
            layer_state: LayerState {
                input_vals: vec![QuantizedTensor::from(x), QuantizedTensor::from(y)],
                output_vals: vec![QuantizedTensor::from(result)],
            },
            advice_value: vec![],
        };

        let virtual_sequence = I::virtual_trace(trace_row);
        assert_eq!(virtual_sequence.len(), I::SEQUENCE_LENGTH);

        for row in virtual_sequence {
            let s1_val = row.layer_state.input_vals[0].clone();
                assert_eq!(
                    registers.get(&row.instruction.input_refs[0]).unwrap(),
                    &s1_val,
                    "{row:?}"
                );
            let rs2_val = row.layer_state.input_vals[1].clone();
                assert_eq!(
                    registers.get(&row.instruction.input_refs[1]).unwrap(),
                    &rs2_val,
                    "{row:?}"
                );

            // let lookup = ONNXInstruction::try_from(&row).unwrap(); 
            let output = unimplemented!(); // lookup.lookup_entry();
            let rd = row.instruction.output_refs[0].clone();
                registers.insert(rd, output);
                assert_eq!(
                    registers.get(&rd).unwrap(),
                    &row.layer_state.output_vals[0],
                    "{row:?}"
                );
        }

        for (key, val) in registers.iter() {
            if key == &r_x.clone() {
                if r_x != rd {
                    // Check that r_x hasn't been clobbered
                    assert_eq!(*val, QuantizedTensor::from(x));
                }
            } else if key == &r_y.clone() {
                if r_y != rd {
                    // Check that r_y hasn't been clobbered
                    assert_eq!(*val, QuantizedTensor::from(y));
                }
            } else if key == &rd.clone() {
                // Check that result was written to rd
                assert_eq!(*val, QuantizedTensor::from(result));
            } 
        }
    }
}