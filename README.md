# When language models know they are wrong

Master's thesis of **Arleen Kaur**, University of California, Riverside.  
Advisor: **Professor Evangelos Papalexakis**.

This repository is the code and experimental record for that thesis. It builds on work by **Het Patel**: the SAE intervention pipeline and the finished uncertainty results below are his, from [*LLMs Encode When They're Wrong: Dissociating Uncertainty and Incorrectness via Sparse Autoencoders*](https://github.com/HettyPatel/sae-uncertainty). The thesis keeps that pipeline and extends it to contextual faithfulness.

The question carried over from his project is whether we can tell, from inside a model, the difference between "the model is unsure" and "the model is confidently wrong" — and whether those signals can be edited.

## Goal

Large language models often sound sure of answers that are false. Uncertainty (the model hedges) and incorrectness (the model is wrong) get treated as the same problem. They are not.

The thesis uses sparse autoencoders (SAEs) to split a model's internal activations into interpretable features, then tests what those features actually do. The aim is a feature-level account of when a model is uncertain, when it is wrong, and — in the work underway now — when it is ignoring the evidence in front of it.

## What is done

These experiments are Het Patel's. They run on **Llama-3.1-8B** (Llama Scope SAEs, 32k features) and **Gemma-2-9B** (Gemma Scope SAEs, 16k features). Questions come from multiple-choice benchmarks (MMLU, RACE, ARC). Each question is scored for correctness and for how peaked the model's answer distribution is (entropy).

That split produces three kinds of features:

- **Uncertainty features** fire when the model is correct but unsure.
- **Incorrectness features** fire when the model is confidently wrong.
- **Confounded features** fire in both cases, so the two signals are mixed.

Suppressing a feature and re-running the model shows what it is causally doing, not just what it correlates with. On Llama-3.1-8B, MMLU validation (7,021 questions; baseline accuracy 61.91%, mean entropy 0.824), suppressing each population has a different effect:

- **Uncertainty features are necessary.** Suppressing them drops accuracy by 13.4%.
- **Incorrectness features are passengers.** Suppressing them changes accuracy by +0.03% and leaves entropy essentially unchanged.
- **Confounded features are harmful.** Suppressing the subset that survives screening (55 of 102) raises accuracy by 1.10% and cuts mean entropy by 0.619, about 75%.

A random-feature control matched to that confounded set moves accuracy by +0.04% and entropy by −0.005, so the gain is not from suppressing arbitrary features.

The same confounded features transfer across datasets: MMLU → ARC (+0.81% accuracy), MMLU → RACE (+0.70%), RACE → MMLU (+0.75%). The dissociation replicates on Gemma-2-9B.

Follow-up checks already in the repo: random-feature controls, screening-criterion ablations, linear probes that predict correctness and entropy from SAE features, selective prediction (abstain when the probe is unsure), and a self-abstention setup that adds an "I don't know" option.

## What is in progress

The current chapter, which is this thesis, moves from multiple-choice confidence to **contextual faithfulness**. The question is whether SAE features can be turned up or down so the model follows a provided passage instead of an answer stored in its weights. It uses Het Patel's SAE editing method on new data.

Evaluation sets are built from FaithEval-Counterfactual (answer in the context vs. the model's memorized answer) and RAGTruth. Each run has a pre-specified success test against frequency-matched random features. Those runs are underway and are not reported here yet.

## How the method works

This is Het Patel's intervention. At a chosen layer, the residual stream is encoded by a pretrained SAE. Target features are zeroed (or scaled). The edit is applied as a delta on the original hidden state, so the SAE's reconstruction error cancels out and only the chosen features change. Discovery and validation use separate question splits. The random seed is 42.

Pipeline, in order:

1. Cache SAE activations for every question and layer.
2. Test each feature (Mann-Whitney U) to label it as uncertainty, incorrectness, or confounded.
3. Suppress features one at a time and keep those that do not hurt accuracy and that lower entropy.
4. Suppress the kept set together on held-out questions.

Implementation notes for the pipeline live in [`experiments/README.md`](experiments/README.md). Model and SAE settings are in [`configs/models.yaml`](configs/models.yaml).

## Skills this project uses

PyTorch, Hugging Face Transformers, sparse autoencoders (SAELens / Llama Scope / Gemma Scope), causal interventions on residual streams, statistical feature selection, and held-out evaluation on MMLU, RACE, ARC, FaithEval, and RAGTruth.

## License

MIT. See [LICENSE](LICENSE).
