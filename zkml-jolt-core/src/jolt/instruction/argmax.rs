use crate::{
    jolt::instruction::{VirtualInstructionSequence, ge::GEInstruction},
    utils::u64_vec_to_i128_iter,
};
use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::{MAX_TENSOR_SIZE, virtual_tensor_index},
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

pub struct ArgMaxInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for ArgMaxInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = (MAX_TENSOR_SIZE - 1) * 5 + 3;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::ArgMax);
        let zero_tensor = || Tensor::from((0..MAX_TENSOR_SIZE).map(|_| 0i128));
        let scalar_tensor = |scalar: u64| {
            let mut value = vec![0; MAX_TENSOR_SIZE];
            value[0] = scalar;
            Some(Tensor::from(u64_vec_to_i128_iter(&value)))
        };

        // Create tensors for each value in the input array
        // For each index i, create a tensor with v[i] as the first element and zeros elsewhere
        // This prepares individual input values for element-wise comparison operations
        let gathered_ts1 = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                let mut tensor = zero_tensor();
                tensor[0] = cycle.memory_state.ts1_val.as_ref().unwrap()[i];
                tensor
            })
            .collect::<Vec<_>>();

        // Create tensors representing possible argmax indices
        // For each index i, create a tensor with i as the first element and zeros elsewhere
        // These will be used to track and select the position of the maximum value
        let indices = (0..MAX_TENSOR_SIZE)
            .map(|i| {
                let mut tensor = zero_tensor();
                tensor[0] = i as i128;
                tensor
            })
            .collect::<Vec<_>>();

        // Virtual registers used in sequence
        let vmax_idx = Some(virtual_tensor_index(0));
        let vmax_val = Some(virtual_tensor_index(1));
        let vxi_val = Some(virtual_tensor_index(2));
        let vxi_idx = Some(virtual_tensor_index(2));
        let vcond = Some(virtual_tensor_index(3));

        // ArgMax operands
        let x = cycle.ts1_vals();

        let mut virtual_trace = vec![];

        // max_idx = 0
        // 	idx0 = Constant(value=[0]) → [1]
        let mut max_idx = 0;
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualConst,
                ts1: None,
                ts2: None,
                ts3: None,
                td: vmax_idx,
                imm: Some(indices[0].clone()),
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(indices[0].clone()),
            },
            advice_value: None,
        });

        // max_val = x[0]
        // max_val = Gather(data=x, indices=idx0, axis=1) → [1]
        let mut max_val = x[0];
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Gather,
                ts1: cycle.instr.ts1,
                ts2: vmax_idx,
                ts3: None,
                td: vmax_val,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
            },
            memory_state: MemoryState {
                ts1_val: cycle.memory_state.ts1_val.clone(),
                ts2_val: Some(indices[0].clone()),
                ts3_val: None,
                td_pre_val: None,
                td_post_val: Some(gathered_ts1[0].clone()),
            },
            advice_value: None,
        });

        for i in 1..MAX_TENSOR_SIZE {
            // idx_i = Constant(value=[i]) → [1]
            let const_idxi = ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::VirtualConst,
                    ts1: None,
                    ts2: None,
                    ts3: None,
                    td: vxi_idx,
                    imm: Some(indices[i].clone()),
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: None,
                    ts2_val: None,
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: Some(indices[i].clone()),
                },
                advice_value: None,
            };
            virtual_trace.push(const_idxi.clone());

            // x[i] = Gather(data=x, indices=idx_i, axis=1)
            let xi = x[i];
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Gather,
                    ts1: cycle.instr.ts1,
                    ts2: vxi_idx,
                    ts3: None,
                    td: vxi_val,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: cycle.memory_state.ts1_val.clone(),
                    ts2_val: Some(indices[i].clone()),
                    ts3_val: None,
                    td_pre_val: None, // TODO(Forpee): I do not think I have to populate this. I will double check
                    td_post_val: Some(gathered_ts1[i].clone()),
                },
                advice_value: None,
            });

            let ge = { GEInstruction::<WORD_SIZE>(xi, max_val).to_lookup_output() };
            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Gte,
                    ts1: vxi_val,
                    ts2: vmax_val,
                    ts3: None,
                    td: vcond,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: Some(gathered_ts1[i].clone()),
                    ts2_val: scalar_tensor(max_val),
                    ts3_val: None,
                    td_pre_val: None,
                    td_post_val: scalar_tensor(ge),
                },
                advice_value: None,
            });

            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Select,
                    ts1: vcond,
                    ts2: vxi_val,
                    ts3: vmax_val,
                    td: vmax_val,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: scalar_tensor(ge), // TODO: I should probably precompute these values before the loop
                    ts2_val: Some(gathered_ts1[i].clone()),
                    ts3_val: scalar_tensor(max_val),
                    td_pre_val: None,
                    td_post_val: scalar_tensor(if ge == 1 { xi } else { max_val }),
                },
                advice_value: None,
            });
            max_val = if ge == 1 { xi } else { max_val };

            virtual_trace.push(ONNXCycle {
                instr: ONNXInstr {
                    address: cycle.instr.address,
                    opcode: ONNXOpcode::Select,
                    ts1: vcond,
                    ts2: vxi_idx,
                    ts3: vmax_idx,
                    td: vmax_idx,
                    imm: None,
                    virtual_sequence_remaining: Some(
                        Self::SEQUENCE_LENGTH - virtual_trace.len() - 1,
                    ),
                    active_output_elements: 1,
                },
                memory_state: MemoryState {
                    ts1_val: scalar_tensor(ge), // TODO: I should probably precompute these values before the loop
                    ts2_val: Some(indices[i].clone()),
                    ts3_val: scalar_tensor(max_idx),
                    td_pre_val: None,
                    td_post_val: scalar_tensor(if ge == 1 { i as u64 } else { max_idx }),
                },
                advice_value: None,
            });
            max_idx = if ge == 1 { i as u64 } else { max_idx };
        }

        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: vmax_idx,
                ts2: None,
                ts3: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
                active_output_elements: 1,
            },
            memory_state: MemoryState {
                ts1_val: scalar_tensor(max_idx),
                ts2_val: None,
                ts3_val: None,
                td_pre_val: cycle.memory_state.td_pre_val.clone(),
                td_post_val: scalar_tensor(max_idx),
            },
            advice_value: None,
        });

        virtual_trace
    }

    /// This function implements the argmax operation, which finds the index of the maximum
    /// value in an array. The algorithm works as follows:
    ///
    /// 1. Initialize max_val to the first element v[0]
    ///
    /// 2. Initialize max_idx to 0 (index of first element)
    ///
    /// 3. For each subsequent element i from 1 to N-1:
    ///    a. Compare current element v[i] with max_val using >= operator
    ///
    ///    b. If v[i] >= max_val (condition c is true):
    ///       - Update max_val to v[i] (select new maximum value)
    ///       - Update max_idx to i (select new maximum index)
    ///
    ///    c. Otherwise, keep current max_val and max_idx unchanged
    ///
    /// 4. Return the index (max_idx) of the maximum element found
    ///
    /// The select operations are conditional assignments that choose between
    /// two values based on the comparison result:
    /// - select(c, a, b) returns 'a' if condition 'c' is true, otherwise 'b'
    ///
    /// Time Complexity: O(N) where N is the array length
    /// Space Complexity: O(1) as only constant extra space is used
    ///
    /// Example:
    /// Input array: [3, 7, 2, 9, 1]
    /// Step 1: max_val = 3, max_idx = 0
    /// Step 2: i=1, v[1]=7 >= 3 → max_val = 7, max_idx = 1
    /// Step 3: i=2, v[2]=2 < 7 → no change
    /// Step 4: i=3, v[3]=9 >= 7 → max_val = 9, max_idx = 3
    /// Step 5: i=4, v[4]=1 < 9 → no change
    /// Result: max_idx = 3 (index of value 9)
    ///
    /// Pseudocode:
    /// ```ignore
    /// max_val = v[0]
    /// max_idx = 0
    /// for i in 1..N-1:
    ///     c = (v[i] >= max_val)              // boolean comparison
    ///     max_val = select(c, v[i], max_val) // conditional value update
    ///     max_idx = select(c, i, max_idx)    // conditional index update
    /// return max_idx
    /// ```
    fn sequence_output(x: Vec<u64>, _: Vec<u64>) -> Vec<u64> {
        let x = x
            .iter()
            .map(|&v| v as u32 as i32 as i64)
            .collect::<Vec<_>>();
        let mut max_idx = 0;
        let mut max_val = x[max_idx];
        for (i, &xi) in x.iter().enumerate().skip(1) {
            let c = xi >= max_val;
            max_val = if c { xi } else { max_val };
            max_idx = if c { i } else { max_idx };
        }
        // Pad ouput tensor to MAX_TENSOR_SIZE
        // The first element is the index of the maximum value
        // The rest are zeros
        // This is to ensure the output tensor has a fixed size
        let mut output = vec![0; MAX_TENSOR_SIZE];
        output[0] = max_idx as u64;
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    #[test]
    fn argmax_virtual_sequence_32() {
        jolt_virtual_sequence_test::<ArgMaxInstruction<32>>(ONNXOpcode::ArgMax);
    }
}
