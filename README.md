# Readout-Visible-Coherence-QNN: Sector Coherence and Trainability in Noisy Equivariant Quantum Neural Networks

*Readout-visible sector coherence controls active-gradient degradation in noisy equivariant quantum neural networks*, by H. Ugail and N. Howard

Symmetry is widely used to structure quantum neural networks against barren plateaus, but a symmetry-preserving circuit is not automatically protected against noise. A channel can commute with the symmetry and still erase the quantum information that carries the gradient. The deployment question is therefore not whether the noise respects the symmetry, but whether it contracts the specific intra-sector coherence mode that the readout actually responds to.

**This repository is a reproducible toolkit that studies how Markovian noise degrades the parameter gradients of a fixed-state U(1)-equivariant brickwork circuit, and identifies the readout-visible aligned sector-coherence rate as the operator-level quantity that controls that degradation.** Every result is produced by a single density-matrix pipeline at fixed circuit geometry, with the noiseless and noisy gradient second moments evaluated under common random numbers so that any measured degradation reflects the physics rather than a sampling artefact. The central response variable is the relative gradient degradation, read against the operator-level coherence-contraction rate of each channel.


<img width="2106" height="815" alt="fig01_activity_map" src="https://github.com/user-attachments/assets/816e6803-315a-42a8-b854-daa0a2df3ebd" />


## What does this measure?

For each noise channel the toolkit reports the relative gradient degradation as a function of the accumulated noise-depth product and the worst-case coherence-contraction rate, together with its regression against that rate. The headline empirical findings are:

1. **Active gradients are confined to the readout light cone.** The noiseless activity map separates active from inactive parameters by a median ratio above `10^27`, with active parameters lying inside the analytic backward light cone of the readout and every parameter outside it at numerical floor.

2. **Finite-noise degradation follows the accumulated coherence-contraction rate.** Across the main single-excitation sweep, the relative degradation obeys a clean power law in the noise-depth product and the worst-case rate, with `alpha = 1.002`, `beta = 0.964` and `R^2 = 0.979` over 96 settings. Restricting to the three symmetry-preserving channels gives `alpha = 1.006`, `beta = 0.889`, `R^2 = 0.978`, and the symmetry-breaking X-error channel is recovered as a held-out out-of-class control at `R^2 = 0.949`.

3. **Sector coherence outperforms standard channel diagnostics.** Pairing the noise-depth product with the coherence-contraction rate organises the degradation more tightly than average infidelity, unitarity loss, purity loss, or a diamond-distance proxy.

4. **The worst-case rate is not by itself the operative variable.** A correlated-dephasing channel with a large worst-case rate (`lambda_coh = 4.5`) but a readout-visible aligned rate near zero produces no measurable degradation, with `|D_M|` held at the numerical floor below `3e-14` at every noise level. This zero-alignment control is the decisive evidence that the aligned rate, not the bare worst-case rate, governs the response.

5. **The law is a single-excitation, restricted-isotropic result.** Outside the single-excitation sector the worst-case scalar organises the degradation less well, with `R^2 = 0.71` at two excitations and `R^2 = 0.89` at three, marking the boundary of the scalar regime rather than a competing law.

The degradation law is robust under heteroscedasticity-consistent standard errors, a cluster bootstrap over channel-depth cells, mixed-effects modelling, a permutation test, and leave-one-factor-out cross-validation, and it carries over to an open-chain topology and across system sizes `n` in `{6, 8, 10}`.

## Contents

- **`run_pipeline_full.ipynb`** — the full reproduction notebook. Builds the equivariant brickwork model, the noise channels and the operator-level coherence-contraction rate, runs the light-cone activity maps, the perturbative small-noise study, the main finite-noise sweep, the predictor comparison, the structured-noise and zero-alignment control, the higher-sector probes, and the size and topology checks, and writes every CSV and figure to `Results/`. Autodetects whether it is running in Google Colab or locally.

- **`verify_results.py`** — a quick verification script for reviewers. Loads the precomputed tables in `Results/` and confirms that every headline number in the paper is reproducible from those tables. Runs in a couple of seconds, needs no GPU and no re-simulation, and prints a clean PASS/FAIL summary.

- **`Results/`** — the precomputed result tables and quick-look figures, sufficient to verify every reported number without re-running the pipeline. The CSVs hold the per-phase breakdowns: the depth sweep, the perturbative micro-noise fits, the leave-one-channel-out regressions, the per-sector and structured-noise results, the predictor comparison, and the robustness diagnostics. `Results/figures/` holds the quick-look figures.

- **`requirements.txt`** — full dependencies for the notebook (NumPy, SciPy, pandas, matplotlib, statsmodels). **`requirements-verify.txt`** — minimal dependencies for `verify_results.py` only (NumPy, pandas).

## Quick start

The fastest way to check the headline numbers is:

```
git clone https://github.com/ugail/Readout-Visible-Coherence-QNN.git
cd Readout-Visible-Coherence-QNN
pip install -r requirements-verify.txt
python verify_results.py
```

The script loads the precomputed tables in `Results/`, re-checks every headline quantity claimed in the manuscript, and prints a PASS/FAIL summary. A full pass takes a couple of seconds on any laptop and needs no quantum simulation.

## Reproducing the full pipeline

Re-running the full pipeline requires Python 3.10 or later. No GPU is needed, though the larger system sizes are memory-intensive because the simulation evolves full density matrices. Install the full requirements first:

```
pip install -r requirements.txt
```

Then open the notebook:

```
jupyter notebook run_pipeline_full.ipynb
```

The notebook autodetects whether it is running in Google Colab or locally. In Colab it can mount Google Drive to write results; locally it writes to a path you set in the first configuration cell. It writes every CSV and figure to `Results/`. Fixed random seeds are used throughout, so every reported quantity is reproducible from the archived data.

## The model in brief

The circuit is a depth-`L` U(1)-equivariant brickwork ansatz on a cycle graph. Each layer applies trainable single-qubit `Z`-rotations on every site, followed by nearest-neighbour `XY` hopping gates on alternating edges; both gate families commute with the total charge, so the circuit is exactly equivariant. The hopping parameters are fixed and treated as part of the architecture, so the trainable freedom is the single-qubit rotation angles. The circuit acts on the fixed localised input `|1^r 0^(n-r)>` in charge sector `r`, and the readout is the sector-projected occupancy of the first qubit.

The object of study is the second moment of the gradient of this readout with respect to the trainable angles, averaged over a small-box parameter prior. For each noise channel the toolkit computes:

- **`lambda_coh`** — the worst-case coherence-contraction rate, the largest decay rate of the restricted noise generator on the intra-sector off-diagonal block, evaluated directly from the channel generator.
- **`lambda_vis`** — the readout-visible aligned rate, the Rayleigh quotient of the restricted dissipator along the gradient-carrying mode. It is bounded above by `lambda_coh`, with equality for restricted-isotropic noise.
- **`D_M`** — the relative gradient degradation, the log ratio of the noiseless to noisy second moment, paired under common random numbers to suppress sampling variance.
- **`omega`** — the alignment factor `lambda_vis / lambda_coh`, in `[0, 1]`, estimated perturbatively and, separately, as a finite-noise empirical weight.

The noiseless and noisy parameter-shift evaluations share random draws, so the degradation ratio is far less sensitive to the sampling floor than independent draws would be. This is what makes the perturbative regime, where the degradation is small, measurable.

## Scope

The clean scalar law is a single-excitation-sector result for the restricted-isotropic channel family, in which the aligned rate reduces to the worst-case scalar. Higher sectors and structured noise call for the aligned diagnostic explicitly, and coherent unitary miscalibration, which rotates rather than contracts off-diagonal modes, sits outside the dissipative family treated here. Full density-matrix simulation limits the accessible system sizes, and the size check reaches `n = 10`. Direct generator-level evaluation of `lambda_vis` across the structured family is left to future work; the present evidence for the aligned rate comes from the zero-alignment correlated-dephasing control and the alignment-weighted refinement.

## Who is this for?

- **Researchers in quantum machine learning** who want a concrete demonstration that symmetry preservation and sector population are insufficient predictors of noisy trainability, and that the readout-visible coherence is the matched quantity.
- **Researchers in open quantum systems** who want a worked operator-level account of how a Markovian channel degrades a parameter gradient, with the worst-case and aligned rates separated.
- **Reviewers** who want to verify every headline number from precomputed tables in a couple of seconds, without re-running the pipeline.

## Citation

If you use this toolkit or the precomputed result tables, please cite:

> H. Ugail and N. Howard, *Readout-visible sector coherence controls active-gradient degradation in noisy equivariant quantum neural networks*, 2026, Under review.

## License

Released under the MIT License. See `LICENSE` for the full text.
