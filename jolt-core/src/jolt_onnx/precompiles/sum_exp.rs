//! A sum-check precompile implementation for softmax operation.
//! Used for proving correctness of the execution of the softmax ONNX operator.
//! You can see it in action in [`crate::jolt_onnx::vm::precompiles`]
//!
//! # Overview:
//!   - [`SoftmaxPrecompile`] - We specify the precompile for softmax op, by defining the input (z) vector.
//!   - [`SoftmaxSumcheck`] - Defines the prover and verifier states that will be used to instantiate a [`super::sumcheck_engine::BatchedSumcheck`] instance.
//!     These sum-check instances are then fed into [`super::sumcheck_engine::BatchedSumcheck::prove`] and [`super::sumcheck_engine::BatchedSumcheck::verify`].
//!   - [`SoftmaxProverState`] - Handles/Defines the prover state for the softmax sum-check precompile (handles witness polynomials for sum-check prover).
//!   - [`SoftmaxVerifierState`] - Handles/Defines the verifier state for the softmax sum-check precompile.

use crate::{
    field::JoltField,
    jolt_onnx::precompiles::sumcheck_engine::BatchableSumcheckInstance,
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, multilinear_polynomial::BindingOrder,
    },
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::Itertools;
use num_traits::Pow;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use ark_ff::Field; 

pub const SCALE: u64 = 256;
/// Input scale for softmax. Input values are between -128 and 127. Quantized input values are between 0 and 255.
pub const INPUT_SCALE: f32 = 1.0 / 256.0;
/// Output scale for softmax. Output values are between 0 and 1. Quantized output values are between 0 and 255.
pub const OUTPUT_SCALE: f32 = 1.0 / 256.0;

/// A type defining the sum_exp precompile in the execution trace.
/// The type is used to intialize the [`SumExpProverState`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SumExpPrecompile {
    z: Vec<u8>,
}

impl SumExpPrecompile {
    /// Create a new instance of [`SumExpPrecompile`].
    pub fn new(z: Vec<u8>) -> Self {
        Self { z }
    }

    /// Returns the maximum value in the input vector.
    pub fn max(&self) -> u8 {
        *self.z.iter().max().unwrap()
    }

    /// Returns the sum of the exponentials of the input vector.
    pub fn execute_sum_exp(&self) -> u64 {
        self.execute_exp().iter().sum()
    }

    /// Execute the softmax operation.
    pub fn execute_exp(&self) -> Vec<u64> {
        let n = self.z.len();
        let max = self.max();
        let mut output = vec![0u64; n];

        for i in 0..n {
            // let z_shifted = self.z[i] as i32 - max as i32;
            let z_shifted = self.z[i];
            let e_z_i = 3.0f32.pow(z_shifted as f32) as u64;
            output[i] = e_z_i;
        }

        output
    }

    fn exp_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        let exp_eval = self.execute_exp();
        DensePolynomial::new(
            exp_eval
                .iter()
                .map(|&x| F::from_u64(x as u64))
                .collect_vec(),
        )
    }

    fn z_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        DensePolynomial::new(self.z.iter().map(|&x| F::from_u64(x as u64)).collect_vec())
    }
}

/// Container type to manage the prover state in the [`BatchableSumcheckInstance`] for the sum_exp precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumExpProverState<F>
where
    F: JoltField,
{
    z: DensePolynomial<F>,
    /// exp(x_i) after the exponential pre-compile
    exp: DensePolynomial<F>,
    /// number of remaining folding rounds
    num_rounds: usize,
    /// initial public claim  Σ_i exp(x_i) 
    input_claim: F,
}

impl<F> SumExpProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SumExpProverState`].
    ///
    /// We apply sum-check to the log(n) variate polynomial Σₖ z(k) * eq(k, r)
    pub fn initialize<ProofTranscript>(
        input: &SumExpPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let n = input.z.len();
        let ri: Vec<F> = transcript.challenge_scalar_powers(n.log_2());

        let exp = input.exp_poly();
        let input_claim = exp.evaluate(&ri);

        println!("input_claim: {}", input_claim);
        transcript.append_scalar(&input_claim);

        let num_rounds = n.log_2();

        Self {
            z: input.z_poly(),
            exp,
            input_claim: input_claim,
            num_rounds,
        }
    }

    fn z_claim(input: &SumExpPrecompile, ri: &[F]) -> F {
        input.z_poly().evaluate(ri)
    }

    fn exp_claim(input: &SumExpPrecompile, ri: &[F]) -> F {
        input.exp_poly().evaluate(ri)
    }
}

/// Dimensions for the softmax inputs.
#[derive(Clone, Serialize, Deserialize, Debug, Copy)]
pub struct SumExpPrecompileDims {
    /// Length of the input vector
    pub n: usize,
}

/// Container type to manage the verifier state in the [`BatchableSumcheckInstance`] for the softmax precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumExpVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> SumExpVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SumExpVerifierState`].
    pub fn initialize<ProofTranscript>(
        dims: SumExpPrecompileDims,
        input_claim: F,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let num_rounds = dims.n.log_2();
        let _ri: Vec<F> = transcript.challenge_scalar_powers(dims.n.log_2());
        transcript.append_scalar(&input_claim);
        Self {
            num_rounds,
            input_claim,
        }
    }
}

/// Stores the evaluations of the s polynomial at `r_i`
/// Where:
///   - `r_i` ∈ F^{log(n)}
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumExpClaims<F>
where
    F: JoltField,
{
    exp: F,
    z: F,
}

/// Batchable sum-check instance for softmax precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SumExpSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<SumExpProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<SumExpVerifierState<F>>,
    /// Holds the final claims for the softmax sum-check precompile.
    pub claims: Option<SumExpClaims<F>>,
}

impl<F> SumExpSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`SoftmaxSumcheck`]
    pub fn new(
        prover_state: Option<SumExpProverState<F>>,
        verifier_state: Option<SumExpVerifierState<F>>,
        claims: Option<SumExpClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for SumExpSumcheck<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().num_rounds
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().input_claim
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().input_claim
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    #[tracing::instrument(skip_all)]
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let SumExpProverState {
            exp, z, ..
        } = self.prover_state.as_ref().unwrap();
        let len = exp.len() / 2; 
        let g0 = (0..len)
            .into_iter()
            .map(|i| {
                let mut g_i = F::one();
                println!("z[i]: {}", z[i]);
                for _ in 0..(z[i].to_u64().unwrap()) {
                    g_i = g_i * F::from_u64(3);
                }
                // F::from_u64(3).pow([z[i].to_u64().unwrap()]);
                println!("i, g_i: {}, {}", i, g_i);
                g_i
            })
            .reduce(|acc, v| acc + v)
            .unwrap_or(F::zero());
        vec![g0]
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let SumExpProverState { z, .. } = self.prover_state.as_mut().unwrap();
        z.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(&mut self) {
        let SumExpProverState {
            exp, z, ..
        } = self.prover_state.as_ref().unwrap();
        self.claims = Some(SumExpClaims {
            exp: exp[0],
            z: z[0],
        });
    }

    /// final check: exp_0 = Σ_i exp(x_i)
    fn expected_output_claim(&self, _: &[F]) -> F {
        let SumExpClaims { exp, z, .. } = self.claims.as_ref().unwrap();
        *exp
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::precompiles::{
            sum_exp::{
                SumExpPrecompile, SumExpPrecompileDims, SumExpProverState, SumExpSumcheck,
                SumExpVerifierState,
            },
            sumcheck_engine::{BatchableSumcheckInstance, BatchedSumcheck},
        },
        utils::transcript::{KeccakTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng};
    use itertools::Itertools;
    use rand_core::RngCore;

    #[test]
    fn test_random_execution_trace() {
        let mut rng = test_rng();
        let trace_length = 10;
        let mut pp: Vec<SumExpPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let n = (rng.next_u32() as usize % 200 + 50).next_power_of_two();
            let z = (0..n)
                .map(|_| rng.gen_range(0..=255) as u8)
                .collect_vec();
            let precompile = SumExpPrecompile::new(z);
            pp.push(SumExpPrecompileDims { n });
            let prover_state = SumExpProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = SumExpSumcheck::new(Some(prover_state), None, None);
            sumcheck_instances.push(sumcheck_instance);
        }
        let init_claims = sumcheck_instances
            .iter()
            .map(|p| p.prover_state.as_ref().unwrap().input_claim)
            .collect_vec();
        let trait_objects: Vec<&mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            sumcheck_instances
                .iter_mut()
                .map(|p| p as &mut dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let (sumcheck_proof, _rsc) = BatchedSumcheck::prove(trait_objects, &mut ptranscript);
        let final_claims = sumcheck_instances
            .iter()
            .map(|p| p.claims.as_ref().unwrap().clone())
            .collect_vec();
        let mut vtranscript = KeccakTranscript::new(b"test");
        let mut vsumcheck_instances = Vec::with_capacity(trace_length);
        for ((dims, init_claim), final_claim) in pp
            .iter()
            .zip_eq(init_claims.iter())
            .zip_eq(final_claims.iter())
        {
            let verifier_state =
                SumExpVerifierState::<Fr>::initialize(*dims, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(SumExpSumcheck::new(
                None,
                Some(verifier_state),
                Some(final_claim.clone()),
            ))
        }
        let trait_objects: Vec<&dyn BatchableSumcheckInstance<Fr, KeccakTranscript>> =
            vsumcheck_instances
                .iter()
                .map(|p| p as &dyn BatchableSumcheckInstance<Fr, KeccakTranscript>)
                .collect();
        let _r = BatchedSumcheck::verify(&sumcheck_proof, trait_objects, &mut vtranscript).unwrap();
    }
}
