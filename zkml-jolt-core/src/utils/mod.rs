/// Helper function to convert Vec<u64> to iterator of i128

// TODO(AntoineF4C5): generic quantization
pub fn u64_vec_to_i128_iter(vec: &[u64]) -> impl Iterator<Item = i128> + '_ {
    vec.iter().map(|v| *v as u32 as i32 as i64 as i128)
}

pub fn u64_vec_to_i32_iter(vec: &[u64]) -> impl Iterator<Item = i32> + '_ {
    vec.iter().map(|v| *v as u32 as i32)
}
