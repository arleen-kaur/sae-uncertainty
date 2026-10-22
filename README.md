# When language models know they are wrong

Master's thesis of **Arleen Kaur**, University of California, Riverside.  
Advisor: **Professor Evangelos Papalexakis**.

## The problem

A language model can be wrong in more than one way. It can be unsure. It can be confidently wrong. Or it can ignore a passage it was just given and answer from memory instead.

Those failures get treated as one problem: the model "hallucinated." They do not look like one problem from inside the network. This thesis asks whether sparse autoencoders (SAEs) can separate those signals into features, and whether turning the features up or down changes what the model does.

## Where this starts

**Het Patel** showed that uncertainty and incorrectness are already separable. Using pretrained SAEs on Llama-3.1-8B and Gemma-2-9B, he split multiple-choice questions (MMLU, RACE, ARC) by correctness and by how peaked the answer distribution is, then suppressed the features that marked each group. On Llama-3.1-8B, MMLU validation (7,021 questions; baseline accuracy 61.91%, mean entropy 0.824):

- Suppressing uncertainty features drops accuracy by 13.4%. The model needs them to answer correctly.
- Suppressing incorrectness features changes accuracy by +0.03%. Those features track errors and do not cause them.
- Suppressing screened confounded features (55 of 102) raises accuracy by 1.10% and cuts mean entropy by 0.619, about 75%. A matched random-feature control moves accuracy by +0.04%.

The same confounded features transfer across datasets (MMLU → ARC +0.81%, MMLU → RACE +0.70%, RACE → MMLU +0.75%) and the split replicates on Gemma-2-9B. That study is [*LLMs Encode When They're Wrong: Dissociating Uncertainty and Incorrectness via Sparse Autoencoders*](https://github.com/HettyPatel/sae-uncertainty). This thesis keeps his intervention pipeline and asks the question he did not: what happens when the model has a source it should follow.

## What this thesis adds

The new setting is **contextual faithfulness**. The model is given a passage whose answer disagrees with what it memorized. A faithful answer follows the passage. A memory answer follows the weights.

The experiments test whether SAE features found on that contrast can be scaled so the model moves toward the passage. Evaluation sets are built from FaithEval-Counterfactual and RAGTruth. Each run has a success test fixed before looking at the held-out split: the effect has to beat frequency-matched random features, not just move in the hoped-for direction. Those runs are underway and are not reported here yet.

Code for this part of the thesis:

- `scripts/create_faitheval_eval_set.py` and `scripts/create_ragtruth_eval_set.py` build the eval sets, with discovery and validation splits.
- `scripts/faith_gap.py` finds passage-following and memory features and intervenes on them.
- `src/model.py` and `src/utils.py` score eval sets with however many answer choices the set actually has.

## How a run works

At a chosen layer, the residual stream is encoded by a pretrained SAE. Target features are zeroed or scaled. The edit is applied as a delta on the original hidden state, so the SAE's reconstruction error cancels out and only the chosen features change. Discovery and validation use separate questions. The random seed is 42.

1. Cache SAE activations for every question and layer.
2. Test each feature and label it.
3. Edit the selected features.
4. Measure the change on held-out questions, against a random-feature control.

Details of the original pipeline are in [`experiments/README.md`](experiments/README.md). Model and SAE settings are in [`configs/models.yaml`](configs/models.yaml).

## Skills this project uses

PyTorch, Hugging Face Transformers, sparse autoencoders (SAELens / Llama Scope / Gemma Scope), causal interventions on residual streams, statistical feature selection, and held-out evaluation on MMLU, RACE, ARC, FaithEval, and RAGTruth.

## License

MIT. See [LICENSE](LICENSE).
