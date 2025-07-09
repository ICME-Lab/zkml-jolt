use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use crate::jolt::instruction::{
    add::ADDInstruction, beq::BEQInstruction, mul::MULInstruction,
    virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction, JoltInstruction,
};
use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow, Operator};
use crate::jolt_onnx::instruction::div::DIVInstruction;
use crate::jolt_onnx::instruction::max::MaxInstruction;
use crate::jolt_onnx::instruction::pow_2::Pow2Instruction;
use crate::jolt_onnx::instruction::VirtualInstructionSequence;
use crate::jolt_onnx::precompiles::sum::SumPrecompile;
use crate::jolt_onnx::tracer::tensor::QuantizedTensor;

pub struct SoftmaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for SoftmaxInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 20; // TODO: This is variable, depending on the shape on the tensor

    fn virtual_trace(trace_row: ONNXTraceRow) -> Vec<ONNXTraceRow> {
        assert_eq!(trace_row.instruction.opcode, Operator::Softmax);

        let mut virtual_trace = vec![];

        // TODO: Is it safe to assume that there is only one input value?
        let input = trace_row.layer_state.input_vals[0].clone();

        let max_val = input.data.iter().enumerate().fold(0, |acc, (i, a)| {
            let max = MaxInstruction::<WORD_SIZE>(acc, *a as u64).lookup_entry();
            virtual_trace.push(ONNXTraceRow {
                instruction: ONNXInstruction {
                    opcode: Operator::Max,
                    attributes: None,
                    input_refs: vec![format!("max_acc_{i}"), format!("max_a_{i}")],
                    output_refs: vec![format!("max_out_{i}")],
                },
                layer_state: LayerState {
                    input_vals: vec![
                        QuantizedTensor::new(vec![1], vec![acc as i8], 1.0),
                        QuantizedTensor::new(vec![1], vec![*a as i8], 1.0),
                    ],
                    output_vals: vec![QuantizedTensor::new(vec![1], vec![max as i8], 1.0)],
                },
                advice_value: vec![],
            });
            max
        });

        let a_vec: Vec<u64> = input
            .data
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let a = MULInstruction::<WORD_SIZE>(*z as u64, 63 as u64).lookup_entry();
                virtual_trace.push(ONNXTraceRow {
                    instruction: ONNXInstruction {
                        opcode: Operator::Mul,
                        attributes: None,
                        input_refs: vec![format!("mul_z_{i}"), format!("mul_63_{i}")],
                        output_refs: vec![format!("mul_a_{i}")],
                    },
                    layer_state: LayerState {
                        input_vals: vec![
                            QuantizedTensor::from(*z as i8),
                            QuantizedTensor::from(63 as i8),
                        ],
                        output_vals: vec![QuantizedTensor::from(a as i8)],
                    },
                    advice_value: vec![],
                });
                let b = DIVInstruction::<WORD_SIZE>::sequence_output(
                    QuantizedTensor::from(a),
                    QuantizedTensor::from(max_val),
                );
                let div_virtual_trace = DIVInstruction::<WORD_SIZE>::virtual_trace(ONNXTraceRow {
                    instruction: ONNXInstruction {
                        opcode: Operator::Div,
                        attributes: None,
                        input_refs: vec![format!("div_a_{i}"), format!("div_max_{i}")],
                        output_refs: vec![format!("div_b_{i}")],
                    },
                    layer_state: LayerState {
                        input_vals: vec![
                            QuantizedTensor::from(a as i8),
                            QuantizedTensor::from(max_val as i8),
                        ],
                        output_vals: vec![b.clone()],
                    },
                    advice_value: vec![],
                });
                virtual_trace.extend(div_virtual_trace);

                let pow_2 = Pow2Instruction(b.data[0] as u64).lookup_entry();
                virtual_trace.push(ONNXTraceRow {
                    instruction: ONNXInstruction {
                        opcode: Operator::Pow2,
                        attributes: None,
                        input_refs: vec![format!("pow_2_b_{i}")],
                        output_refs: vec![format!("pow_2_out_{i}")],
                    },
                    layer_state: LayerState {
                        input_vals: vec![b],
                        output_vals: vec![QuantizedTensor::from(pow_2 as i8)],
                    },
                    advice_value: vec![],
                });

                pow_2
            })
            .collect();

        // 4. Run the sum-check to prove that $\sum a_i = N$, where $N$ is the normalisation factor.
        let sum = a_vec.iter().enumerate().fold(0, |acc, (i, &a)| {
            let new_acc = ADDInstruction::<WORD_SIZE>(acc, a as u64).lookup_entry();
            virtual_trace.push(ONNXTraceRow {
                instruction: ONNXInstruction {
                    opcode: Operator::Add,
                    attributes: None,
                    input_refs: vec![format!("sum_acc_{i}"), format!("sum_a_{i}")],
                    output_refs: vec![format!("sum_out_{i}")],
                },
                layer_state: LayerState {
                    input_vals: vec![
                        QuantizedTensor::from(acc as i8),
                        QuantizedTensor::from(a as i8),
                    ],
                    output_vals: vec![QuantizedTensor::from(new_acc as i8)],
                },
                advice_value: vec![],
            });
            new_acc
        });
        // 5. Run a division lookup $a_i / N$ for each element in $\vec{a}$. Since the value is in (0,1), we multiply by $2^8$ for quantization.
        a_vec.iter().enumerate().for_each(|(i, &a)| {
            let b = DIVInstruction::<WORD_SIZE>::sequence_output(
                QuantizedTensor::from(a),
                QuantizedTensor::from(sum),
            );
            let div_virtual_trace = DIVInstruction::<WORD_SIZE>::virtual_trace(ONNXTraceRow {
                instruction: ONNXInstruction {
                    opcode: Operator::Div,
                    attributes: None,
                    input_refs: vec![format!("div_a_{i}"), format!("div_sum_{i}")],
                    output_refs: vec![format!("div_b_{i}")],
                },
                layer_state: LayerState {
                    input_vals: vec![
                        QuantizedTensor::from(a as i8),
                        QuantizedTensor::from(sum as i8),
                    ],
                    output_vals: vec![b],
                },
                advice_value: vec![],
            });
            virtual_trace.extend(div_virtual_trace);
        });

        virtual_trace
    }

    fn sequence_output(x: QuantizedTensor, _y: QuantizedTensor) -> QuantizedTensor {
        /// 1. Take the maximum element from $[z_1, ..., z_n]$. Call this element $z_{max}$. We can recursively apply the [max instruction](https://github.com/ICME-Lab/zkml-jolt/pull/12) as $z_{max} = max(...(max(max(z_0, z_1), z_2),..., z_n)$.
        let max_val = x.data.iter().max().unwrap();
        // 2. Multiply each element by $63$ and divide it by $z_{max}$, that is, $z'i = z_i * 63 / z_{max}$ so that the maximum element is now to $63$ and thus no element $2^{z'_i}$ overflows. The sum $\sum 2^{z'_i}$ must not overflow either. This is a more restrictive form of quantization.
        let a_vec: Vec<u64> = x
            .data
            .iter()
            .map(|z| {
                let normalized = *z as u64 * 63 / *max_val as u64;
                // 3. Compute the "power-of-two" lookup table for each $z'_i$ (i.e., $2^{z'_i}$). This will return a vector $\vec{a} = [2^{z'_1},..., 2^{z'_n}]$.
                let pow_2 = 1 << (normalized as usize);
                pow_2
            })
            .collect();
        // 4. Run the sum-check to prove that $\sum a_i = N$, where $N$ is the normalisation factor.
        let a_sum: u64 = a_vec.iter().sum::<u64>();

        // 5. Run a division lookup $a_i / N$ for each element in $\vec{a}$. Since the value is in (0,1), we multiply by $2^8$ for quantization.
        let data: Vec<i8> = a_vec
            .iter()
            .map(|a| ((*a as f32 / a_sum as f32) * 128.0) as i8)
            .collect();
        QuantizedTensor::new(vec![1], data, 1.0)
    }
}

#[cfg(test)]
mod test {

    use crate::jolt_onnx::instruction::test::jolt_onnx_virtual_sequence_test;

    use super::*;

    #[test]
    fn softmax_virtual_sequence() {
        jolt_onnx_virtual_sequence_test::<SoftmaxInstruction<32>>(Operator::Softmax);
    }
}
