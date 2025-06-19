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
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Input scale for softmax. Input values are between -128 and 127. Quantized input values are between 0 and 255.
pub const INPUT_SCALE: f32 = 1.0 / 256.0;
/// Output scale for softmax. Output values are between 0 and 1. Quantized output values are between 0 and 255.
pub const OUTPUT_SCALE: f32 = 1.0 / 256.0;

/// A type defining the softmax precompile in the execution trace.
/// The type is used to intialize the [`SoftmaxProverState`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SoftmaxPrecompile {
    z: Vec<i8>,
}

impl SoftmaxPrecompile {
    /// Create a new instance of [`SoftmaxPrecompile`].
    pub fn new(z: Vec<i8>) -> Self {
        Self { z }
    }

    /// Returns the maximum value in the input vector.
    pub fn max(&self) -> i8 {
        *self.z.iter().max().unwrap()
    }


    /// Returns the evaluations polynomial `s` of the Softmax operation
    /// s(i) = exp(z[i] - max) / Σ_{j=0}^{n-1} exp(z[j] - max)
    ///
    /// Used to compute the input claim s(r).
    fn s_poly<F>(&self) -> DensePolynomial<F>
    where
        F: JoltField,
    {
        let s_eval = self.execute_softmax();
        DensePolynomial::new(s_eval.iter().map(|&x| F::from_i64(x as i64)).collect_vec())
    }

    /// Execute the softmax operation.
    pub fn execute_softmax(&self) -> Vec<i32> {
        let n = self.z.len();
        let max = self.max();
        let mut output = vec![0i32; n];
        let mut output_f32 = vec![0f32; n];

        let mut normalized_sum = 0f32;
        for i in 0..n {
            // Shift the input by the max value.
            let z_shifted = (self.z[i] as i64 - max as i64) as f32; 
            let e_z_i = z_shifted.exp();
            output_f32[i] = e_z_i;
            normalized_sum += e_z_i;
        }

        for i in 0..n {
            let res = (output_f32[i] / normalized_sum) as f32;
            let res_requant = (res / OUTPUT_SCALE).round();
            output[i] = res_requant as i32; 

        }

        output
    }
}

/// Container type to manage the prover state in the [`BatchableSumcheckInstance`] for the softmax precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SoftmaxProverState<F>
where
    F: JoltField,
{
    /// Softmax multilinear polynomial
    pub s: DensePolynomial<F>,
    /// Evaluation at point r for the softmax multilinear polynomial s
    pub input_claim: F,
    /// Number of rounds in the sum-check precompile
    pub num_rounds: usize,
}

impl<F> SoftmaxProverState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SoftmaxProverState`].
    ///
    /// We apply sum-check to the log(n) variate polynomial Σₖ z(k) * eq(k, r)
    pub fn initialize<ProofTranscript>(
        input: &SoftmaxPrecompile,
        transcript: &mut ProofTranscript,
    ) -> Self
    where
        ProofTranscript: Transcript,
    {
        let n = input.z.len();
        let ri: Vec<F> = transcript.challenge_scalar_powers(n.log_2());
        let s_r = Self::s_mle(&input.execute_softmax().iter().map(|&x| F::from_i64(x as i64)).collect_vec(), &ri);


        let input_claim = Self::input_claim(input, &ri);
        transcript.append_scalar(&input_claim);

        let num_rounds = n.log_2();

        #[cfg(test)]
        {
            let sum: F = s_r.Z.iter().sum();
            assert_eq!(sum, input_claim)
        }
        Self {
            s: s_r,
            input_claim,
            num_rounds,
        }
    }

    /// Given the challenge vectors compute s(ri)
    fn input_claim(input: &SoftmaxPrecompile, ri: &[F]) -> F {
        input.s_poly().evaluate(ri)
    }

    /// Compute the boolean evaluations for the polynomial s(r).
    /// Used as input to the softmax sum-check-precompile protocol.
    ///
    /// Bounds the input polynomial s as follows:
    ///
    ///     s(rᵢ) = ∑_{u}^{n} eq(u, rᵢ) · s(u)
    fn s_mle(z: &[F], ri: &[F]) -> DensePolynomial<F> {
        let n = z.len();
        let mut s_r = vec![F::zero(); n];
        let eq_ri_evals = EqPolynomial::evals(ri);
        for i in 0..n {
            s_r[i] += eq_ri_evals[i] * z[i];
        }
        DensePolynomial::new(s_r)
    }
}

/// Dimensions for the softmax inputs.
#[derive(Clone, Serialize, Deserialize, Debug, Copy)]
pub struct SoftmaxPrecompileDims {
    /// Length of the input vector
    pub n: usize,
}

/// Container type to manage the verifier state in the [`BatchableSumcheckInstance`] for the softmax precompile.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SoftmaxVerifierState<F>
where
    F: JoltField,
{
    num_rounds: usize,
    input_claim: F,
}

impl<F> SoftmaxVerifierState<F>
where
    F: JoltField,
{
    #[tracing::instrument(skip_all)]
    /// Create a new instance of [`SoftmaxVerifierState`].
    pub fn initialize<ProofTranscript>(
        dims: SoftmaxPrecompileDims,
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

/// Store the final claims/openings to later prove the openings.
/// The final claims for the softmax sum-check precompile.
///
/// Stores the evaluations of the s polynomial at `r_i`
/// Where:
///   - `r_i` ∈ F^{log(n)}
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SoftmaxClaims<F>
where
    F: JoltField,
{
    s: F,
}

/// Batchable sum-check instance for softmax precompile.
/// Used to construct the [`PrecompileProof`] by passing in these instances into [`BatchedSumcheck`].
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, Serialize, Deserialize)]
pub struct SoftmaxSumcheck<F>
where
    F: JoltField,
{
    /// Handles state for prover portion of the sum-check protocol.
    pub prover_state: Option<SoftmaxProverState<F>>,
    /// Handles state for verifier portion of the sum-check protocol.
    pub verifier_state: Option<SoftmaxVerifierState<F>>,
    /// Holds the final claims for the softmax sum-check precompile.
    pub claims: Option<SoftmaxClaims<F>>,
}

impl<F> SoftmaxSumcheck<F>
where
    F: JoltField,
{
    /// Create a new instance of [`SoftmaxSumcheck`]
    pub fn new(
        prover_state: Option<SoftmaxProverState<F>>,
        verifier_state: Option<SoftmaxVerifierState<F>>,
        claims: Option<SoftmaxClaims<F>>,
    ) -> Self {
        Self {
            prover_state,
            verifier_state,
            claims,
        }
    }
}

impl<F, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript> for SoftmaxSumcheck<F>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    #[inline(always)]
    fn degree(&self) -> usize {
        2
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
        let SoftmaxProverState { s, .. } = self.prover_state.as_ref().unwrap();
        let len = s.len() / 2;
        let univariate_poly_evals: [F; 2] = (0..len)
            .into_par_iter()
            .map(|i| {
                let poly_S_bound_point = s[i + len] + s[i + len] - s[i];
                [s[i], poly_S_bound_point]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all)]
    fn bind(&mut self, r_j: F, _: usize) {
        let SoftmaxProverState { s, .. } = self.prover_state.as_mut().unwrap();
        s.bind_parallel(r_j, BindingOrder::HighToLow)
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.claims.is_none());
        let SoftmaxProverState { s, .. } = self.prover_state.as_ref().unwrap();
        self.claims = Some(SoftmaxClaims { s: s[0] });
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        let SoftmaxClaims { s } = self.claims.as_ref().unwrap();
        *s
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        jolt_onnx::precompiles::{
            softmax::{
                SoftmaxPrecompile, SoftmaxPrecompileDims, SoftmaxProverState, SoftmaxSumcheck,
                SoftmaxVerifierState,
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
        let mut pp: Vec<SoftmaxPrecompileDims> = Vec::with_capacity(trace_length);
        let mut ptranscript = KeccakTranscript::new(b"test");
        let mut sumcheck_instances = Vec::with_capacity(trace_length);
        for _ in 0..trace_length {
            let n = (rng.next_u32() as usize % 200 + 50).next_power_of_two();
            let z = (0..n).map(|_| rng.gen_range(-128..=127) as i8).collect_vec();
            let precompile = SoftmaxPrecompile::new(z);
            pp.push(SoftmaxPrecompileDims { n });
            let prover_state = SoftmaxProverState::<Fr>::initialize(&precompile, &mut ptranscript);
            let sumcheck_instance = SoftmaxSumcheck::new(Some(prover_state), None, None);
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
                SoftmaxVerifierState::<Fr>::initialize(*dims, *init_claim, &mut vtranscript);
            vsumcheck_instances.push(SoftmaxSumcheck::new(
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
