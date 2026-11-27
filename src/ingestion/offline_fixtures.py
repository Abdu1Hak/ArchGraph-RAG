"""Deterministic offline fixtures for CI, benchmarks and `--offline-demo` (Phase 1B).

Two pieces:
1. `SYNTHETIC_SEED_MARKDOWN`: short, hand-written Markdown stand-ins for seed papers. They are
   clearly labeled synthetic and are NEVER written into the real Markdown cache.
2. `RecordedExtractionLLM`: a stand-in for Gemini that replays recorded triples. A recorded
   triple is returned only when its `evidence_quote` occurs verbatim in the chunk text sent to
   the "model", so provenance (`source_chunk_id` -> chunk text containing the quote) is real.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SYNTHETIC_BANNER = "> SYNTHETIC OFFLINE FIXTURE: hand-written stand-in for CI, not the original paper text."

SYNTHETIC_SEED_MARKDOWN: Dict[str, str] = {
    "vaswani_2017_transformer": f"""# Attention Is All You Need

{SYNTHETIC_BANNER}

## Abstract

We propose the Transformer, a model architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## 3 Model Architecture

### 3.2 Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

$$
\\mathrm{{MultiHead}}(Q, K, V) = \\mathrm{{Concat}}(head_1, \\ldots, head_h) W^O
$$

### 3.5 Positional Encoding

The Transformer adopts sinusoidal positional encodings because the model contains no recurrence.
""",
    "ainslie_2023_gqa": f"""# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

{SYNTHETIC_BANNER}

## Abstract

Multi-query attention (MQA) uses only a single key-value head and drastically speeds up decoder inference.
We introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate number of key-value heads.

## 2 Method

### 2.2 Grouped-Query Attention

Grouped-query attention extends multi-head attention by sharing each key-value head across a group of query heads.

GQA reduces the KV cache memory bandwidth bottleneck during autoregressive decoding.

| Attention | KV heads | Quality |
| --- | --- | --- |
| MHA | $H$ | Baseline |
| GQA | $G$ | Close to MHA |
| MQA | $1$ | Degraded |
""",
    "deepseek_2024_v2": f"""# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

{SYNTHETIC_BANNER}

## Abstract

We present DeepSeek-V2, a strong Mixture-of-Experts language model characterized by economical training and efficient inference.

## 2 Architecture

### 2.1 Multi-Head Latent Attention

Multi-head Latent Attention (MLA) extends the KV-sharing idea of Grouped-Query Attention (GQA) by jointly compressing keys and values into a low-rank latent vector.

$$
c_t^{{KV}} = W^{{DKV}} h_t
$$

Compared with GQA, MLA reduces the KV cache by 93.3% while achieving stronger performance than MHA.

### 2.2 Decoupled Rotary Position Embedding

Decoupled RoPE extends Rotary Position Embedding (RoPE) so that positional information survives low-rank key-value compression.

### 2.3 DeepSeekMoE

DeepSeek-V2 adopts Multi-head Latent Attention in every attention layer and the DeepSeekMoE architecture in its feed-forward networks.
""",
}


def synthesize_generic_markdown(paper_id: str, title: str, year: int, pillar: str) -> str:
    """Deterministic placeholder for papers without a hand-written fixture."""
    return (
        f"# {title}\n\n{SYNTHETIC_BANNER}\n\n## Abstract\n\n"
        f"This synthetic placeholder stands in for '{title}' ({year}), a {pillar.replace('_', '-')} paper "
        f"in the corpus (paper_id: {paper_id}). Run without --offline-demo to fetch the real PDF from arXiv.\n"
    )


def _rel(
    source: str, source_type: str, relation: str, target: str, target_type: str,
    quote: str, context: Optional[str] = None, confidence: float = 0.93,
) -> Dict[str, Any]:
    return {
        "source_name": source, "source_type": source_type, "relation_type": relation,
        "target_name": target, "target_type": target_type, "evidence_quote": quote,
        "model_or_metric_context": context, "confidence": confidence,
    }


RECORDED_RELATIONSHIPS: Dict[str, List[Dict[str, Any]]] = {
    "vaswani_2017_transformer": [
        _rel("Attention Is All You Need", "Paper", "INTRODUCES", "Transformer", "ModelFamily",
             "We propose the Transformer, a model architecture based solely on attention mechanisms"),
        _rel("Attention Is All You Need", "Paper", "INTRODUCES", "Multi-Head Attention", "ArchComponent",
             "Multi-head attention allows the model to jointly attend to information from different representation subspaces"),
    ],
    "ainslie_2023_gqa": [
        _rel("GQA", "Paper", "INTRODUCES", "Grouped-Query Attention (GQA)", "ArchComponent",
             "We introduce grouped-query attention (GQA), a generalization of multi-query attention"),
        _rel("grouped query attention", "ArchComponent", "EXTENDS", "multi-head attention", "ArchComponent",
             "Grouped-query attention extends multi-head attention by sharing each key-value head across a group of query heads."),
        _rel("GQA", "ArchComponent", "MITIGATES", "KV Cache Bottleneck", "Bottleneck",
             "GQA reduces the KV cache memory bandwidth bottleneck during autoregressive decoding."),
    ],
    "deepseek_2024_v2": [
        _rel("DeepSeek-V2", "Paper", "INTRODUCES", "DeepSeek-V2", "ModelFamily",
             "We present DeepSeek-V2, a strong Mixture-of-Experts language model"),
        _rel("Multi-head Latent Attention (MLA)", "ArchComponent", "EXTENDS", "Grouped-Query Attention (GQA)", "ArchComponent",
             "Multi-head Latent Attention (MLA) extends the KV-sharing idea of Grouped-Query Attention (GQA)"),
        _rel("MLA", "ArchComponent", "MITIGATES", "Key-Value Cache Memory Footprint", "Bottleneck",
             "Compared with GQA, MLA reduces the KV cache by 93.3% while achieving stronger performance than MHA.",
             context="-93.3% KV cache"),
        _rel("Decoupled RoPE", "ArchComponent", "EXTENDS", "Rotary Position Embedding (RoPE)", "ArchComponent",
             "Decoupled RoPE extends Rotary Position Embedding (RoPE) so that positional information survives"),
        _rel("DeepSeek-V2", "ModelFamily", "ADOPTS_COMPONENT", "Multi-head Latent Attention", "ArchComponent",
             "DeepSeek-V2 adopts Multi-head Latent Attention in every attention layer"),
        _rel("DeepSeek-V2", "ModelFamily", "ADOPTS_COMPONENT", "DeepSeekMoE", "ArchComponent",
             "the DeepSeekMoE architecture in its feed-forward networks"),
    ],
}


# ---------------------------------------------------------------------------
# Phase 2 additions: fixtures for the remaining 9 seed papers so retrieval has content to rank.
# Facts are paraphrased from the papers; the banner still marks every document as synthetic.
# ---------------------------------------------------------------------------


def _fx(body: str) -> str:
    return body.replace("{BANNER}", SYNTHETIC_BANNER)


SYNTHETIC_SEED_MARKDOWN.update({
    "hoffmann_2022_chinchilla": _fx("""# Training Compute-Optimal Large Language Models

{BANNER}

## Abstract

We find that for compute-optimal training, model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled.

## 3 Estimating the Optimal Parameter/Token Allocation

Chinchilla uses 70B parameters trained on 1.4 trillion tokens, roughly 20 tokens per parameter, and outperforms Gopher (280B) with the same compute budget.
"""),
    "touvron_2023_llama1": _fx("""# LLaMA: Open and Efficient Foundation Language Models

{BANNER}

## 2.2 Architecture

LLaMA adopts the SwiGLU activation function in place of ReLU in the feed-forward network, using a hidden dimension of 2/3 4d.

LLaMA applies pre-normalization with RMSNorm on the input of each transformer sub-layer.

LLaMA replaces absolute positional embeddings with rotary positional embeddings (RoPE) at each layer of the network.

## 3 Results

LLaMA-13B outperforms GPT-3 (175B) on most benchmarks despite being more than 10x smaller.
"""),
    "dubey_2024_llama3": _fx("""# The Llama 3 Herd of Models

{BANNER}

## 3.2 Model Architecture

Llama 3 uses grouped query attention (GQA) with 8 key-value heads to improve inference speed and reduce the size of key-value caches during decoding.

We increase the RoPE base frequency hyperparameter to 500,000, which enables better support for longer contexts.

Llama 3 uses a vocabulary with 128K tokens, and the flagship model has 405B parameters trained on about 15.6T tokens.
"""),
    "deepseek_2024_v3": _fx("""# DeepSeek-V3 Technical Report

{BANNER}

## Abstract

DeepSeek-V3 is a Mixture-of-Experts language model with 671B total parameters, of which 37B are activated for each token.

## 2 Architecture

DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture, both validated in DeepSeek-V2.

DeepSeek-V3 pioneers an auxiliary-loss-free load balancing strategy that adds a bias term to each expert's routing score instead of an auxiliary loss.

## 3 Infrastructure

We validate FP8 mixed precision training on an extremely large-scale model for the first time.
"""),
    "su_2021_rope": _fx("""# RoFormer: Enhanced Transformer with Rotary Position Embedding

{BANNER}

## 3 Rotary Position Embedding

Rotary Position Embedding (RoPE) encodes the absolute position with a rotation matrix and incorporates explicit relative position dependency in the self-attention formulation.

$$
f_q(x_m, m) = R^d_{\\Theta, m} W_q x_m, \\quad \\theta_i = 10000^{-2(i-1)/d}
$$

RoPE has long-term decay: the inner product between tokens decays as the relative distance increases.
"""),
    "peng_2023_yarn": _fx("""# YaRN: Efficient Context Window Extension of Large Language Models

{BANNER}

## Abstract

YaRN is a compute-efficient method to extend the context window of RoPE-based models, requiring 10x fewer tokens and 2.5x fewer training steps than previous methods.

## 3 Method

YaRN combines NTK-by-parts interpolation of the rotary frequencies with an attention temperature scaling of sqrt(1/t) = 0.1 ln(s) + 1.

YaRN extends the context window of LLaMA models to 128k tokens.
"""),
    "rafailov_2023_dpo": _fx("""# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

{BANNER}

## Abstract

Direct Preference Optimization (DPO) optimizes the policy with a simple binary cross-entropy objective, eliminating the need to fit an explicit reward model or to sample from the policy during fine-tuning.

## 4 Method

$$
L_{DPO} = -\\mathbb{E} [ \\log \\sigma ( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{ref}(y_l|x)} ) ]
$$

DPO replaces the PPO-based RLHF pipeline with a single supervised classification loss on preference pairs.
"""),
    "deepseek_2025_r1": _fx("""# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

{BANNER}

## 2.1 Reinforcement Learning Algorithm

We adopt Group Relative Policy Optimization (GRPO), which foregoes the critic model that is typically the same size as the policy model and estimates the baseline from group scores instead.

## 2.2 DeepSeek-R1-Zero

DeepSeek-R1-Zero is trained via large-scale reinforcement learning without supervised fine-tuning; its AIME 2024 pass@1 score rises from 15.6% to 71.0%.
"""),
    "kwon_2023_pagedattention": _fx("""# Efficient Memory Management for Large Language Model Serving with PagedAttention

{BANNER}

## Abstract

Existing serving systems waste 60% to 80% of KV cache memory because of fragmentation and over-reservation.

## 4 Method

PagedAttention partitions the KV cache of each sequence into fixed-size blocks that need not be contiguous in memory, inspired by virtual memory paging in operating systems.

PagedAttention mitigates the KV cache memory bottleneck so that vLLM achieves near-zero waste and 2-4x higher serving throughput.
"""),
})

RECORDED_RELATIONSHIPS.update({
    "touvron_2023_llama1": [
        _rel("LLaMA", "ModelFamily", "ADOPTS_COMPONENT", "SwiGLU", "ArchComponent",
             "LLaMA adopts the SwiGLU activation function in place of ReLU"),
        _rel("LLaMA", "ModelFamily", "ADOPTS_COMPONENT", "RMSNorm", "ArchComponent",
             "LLaMA applies pre-normalization with RMSNorm on the input of each transformer sub-layer."),
        _rel("LLaMA", "ModelFamily", "ADOPTS_COMPONENT", "Rotary Positional Embeddings (RoPE)", "ArchComponent",
             "LLaMA replaces absolute positional embeddings with rotary positional embeddings (RoPE)"),
    ],
    "dubey_2024_llama3": [
        _rel("Llama 3", "ModelFamily", "ADOPTS_COMPONENT", "Grouped Query Attention (GQA)", "ArchComponent",
             "Llama 3 uses grouped query attention (GQA) with 8 key-value heads"),
    ],
    "deepseek_2024_v3": [
        _rel("DeepSeek-V3", "ModelFamily", "ADOPTS_COMPONENT", "Multi-head Latent Attention (MLA)", "ArchComponent",
             "DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture"),
        _rel("DeepSeek-V3", "ModelFamily", "ADOPTS_COMPONENT", "Auxiliary-Loss-Free Load Balancing", "ArchComponent",
             "DeepSeek-V3 pioneers an auxiliary-loss-free load balancing strategy"),
    ],
    "peng_2023_yarn": [
        _rel("YaRN", "TrainingMethod", "EXTENDS", "Rotary Position Embedding", "ArchComponent",
             "YaRN is a compute-efficient method to extend the context window of RoPE-based models"),
    ],
    "deepseek_2025_r1": [
        _rel("Group Relative Policy Optimization (GRPO)", "TrainingMethod", "MITIGATES", "RLHF Critic Value Model Overhead",
             "Bottleneck", "We adopt Group Relative Policy Optimization (GRPO), which foregoes the critic model"),
    ],
    "kwon_2023_pagedattention": [
        _rel("PagedAttention", "ArchComponent", "MITIGATES", "KV Cache Memory Bandwidth Bottleneck", "Bottleneck",
             "PagedAttention mitigates the KV cache memory bottleneck so that vLLM achieves near-zero waste"),
    ],
})

# ---------------------------------------------------------------------------
# Phase 5 corpus enrichment: multi-paragraph technical detail and caveats for all 12 seed papers.
# Sections are APPENDED after the existing ones so every earlier chunk_id stays stable. Facts are
# paraphrased from the papers. Recorded triples below cover every relation the new text states
# explicitly (one extraction rule for all text, not chosen per benchmark question).
# ---------------------------------------------------------------------------

PHASE5_APPENDIX: Dict[str, str] = {
    "vaswani_2017_transformer": r"""
## 3.3 Scaled Dot-Product Attention

We compute attention as a weighted sum of the values, where each weight is the softmax of a query-key dot product divided by the square root of the key dimension d_k.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(Q K^T / \sqrt{d_k}) V
$$

We scale the dot products by 1/sqrt(d_k) because for large d_k the dot products become large in magnitude and push the softmax into regions with extremely small gradients.

The base model uses h = 8 parallel attention heads with d_k = d_v = d_model / h = 64, so the total cost is similar to single-head attention with full dimensionality.

## 4 Why Self-Attention

A self-attention layer connects all positions with a constant number of sequential operations, whereas a recurrent layer requires O(n) sequential operations.

The complexity of self-attention per layer is $O(n^2 \cdot d)$ in the sequence length n, which becomes the dominant cost for very long sequences.

## 6 Results

The big Transformer reaches 28.4 BLEU on the WMT 2014 English-to-German translation task, more than 2 BLEU above the previous best models including ensembles.
""",
    "ainslie_2023_gqa": r"""
## 2.3 Uptraining Existing Checkpoints

Existing multi-head checkpoints are converted into multi-query or grouped-query checkpoints by mean-pooling the key and value projection heads within each group.

The converted checkpoint is then uptrained with only 5% of the original pre-training compute, which is far cheaper than training a grouped-query model from scratch.

## 2.4 Where GQA Is Applied

GQA is applied to decoder self-attention and cross-attention only; the encoder is left unchanged because encoder representations are computed in parallel and are not limited by memory bandwidth.

## 4 Results

Uptrained GQA-8 reaches quality close to multi-head attention while running almost as fast as multi-query attention.

Multi-query attention can cause quality degradation and training instability, which is why an intermediate number of groups is preferred.
""",
    "deepseek_2024_v2": r"""
## 2.4 Why RoPE Must Be Decoupled

Standard RoPE is position-sensitive for both keys and queries, so the key up-projection matrix W^{UK} could no longer be absorbed into the query projection during inference; MLA therefore carries positional information in an additional shared key k_t^R.

MLA also compresses the queries into a low-rank latent to reduce activation memory during training, but query compression does not shrink the KV cache.

## 3 Efficiency

Compared with DeepSeek 67B, DeepSeek-V2 saves 42.5% of training costs and boosts the maximum generation throughput to 5.76 times.

DeepSeek-V2 contains 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens.

## 4 Alignment

DeepSeek-V2 Chat is aligned with supervised fine-tuning followed by reinforcement learning with GRPO.
""",
    "deepseek_2024_v3": r"""
## 2.2 Multi-Token Prediction

DeepSeek-V3 adds a multi-token prediction (MTP) training objective that predicts several future tokens at each position, densifying the training signal.

The MTP modules are discarded at inference time, or they can be reused as draft heads for speculative decoding.

## 2.3 Load Balancing Details

The routing bias is only used to select experts; the gating value that multiplies each expert output is still computed from the original affinity score.

A tiny sequence-wise balance loss is kept only to avoid extreme imbalance within a single sequence.

## 4 Training Cost

The full training of DeepSeek-V3 takes 2.788M H800 GPU hours, and its pre-training corpus contains 14.8T tokens.

## 5 Post-Training

During post-training, reasoning capability is distilled from the DeepSeek-R1 series of models into DeepSeek-V3.
""",
    "deepseek_2025_r1": r"""
## 2.3 Reward Modeling

DeepSeek-R1-Zero is optimized with GRPO using rule-based accuracy and format rewards rather than a neural reward model, because neural reward models are prone to reward hacking at scale.

PPO, by contrast, requires a separate value model (the critic) of similar size to the policy, which adds substantial memory and compute burden.

## 2.4 Drawbacks of DeepSeek-R1-Zero

DeepSeek-R1-Zero suffers from poor readability and language mixing in its reasoning traces.

## 3 DeepSeek-R1: Reinforcement Learning with Cold Start

Unlike DeepSeek-R1-Zero, DeepSeek-R1 first fine-tunes the base model on thousands of long chain-of-thought cold-start examples before reasoning-oriented reinforcement learning.

Both DeepSeek-R1-Zero and DeepSeek-R1 start from the DeepSeek-V3-Base checkpoint.

## 4 Distillation

Distilling DeepSeek-R1 outputs into smaller Qwen and Llama models gives them strong reasoning without running any reinforcement learning on the small models.
""",
    "rafailov_2023_dpo": r"""
## 2 Background: RLHF with PPO

Standard RLHF first fits a reward model to human preference data and then optimizes the language model with PPO against that learned reward, while a KL penalty keeps the policy close to a reference model.

## 5 Theoretical Notes

DPO still needs a frozen reference policy: the log-probability ratios against the reference model appear directly in its loss, and beta controls how far the policy may drift from it.

## 6 Experiments

On TL;DR summarization and Anthropic HH single-turn dialogue, DPO matches or exceeds PPO-based RLHF while being substantially simpler to train.
""",
    "kwon_2023_pagedattention": r"""
## 4.4 Sharing Across Sequences

PagedAttention lets parallel sampling and beam search candidates share KV blocks with copy-on-write, cutting their memory usage by up to 55%.

## 5 Implementation

vLLM is an LLM serving engine built on top of PagedAttention; because it changes only how the KV cache is stored, model outputs are identical to standard attention.

PagedAttention is complementary to GQA: grouped-query attention shrinks the KV cache per token, whereas PagedAttention removes fragmentation in how that cache is laid out in memory.
""",
    "hoffmann_2022_chinchilla": r"""
## 4 Discussion

Gopher, GPT-3 and other large models of the time were substantially undertrained: their parameter counts were increased much faster than their training data.

Chinchilla reaches an average accuracy of 67.6% on the MMLU benchmark, more than 7 points above Gopher.

Because Chinchilla is 4x smaller than Gopher, it also uses substantially less compute for fine-tuning and inference.
""",
    "touvron_2023_llama1": r"""
## 2.1 Pre-training Data

LLaMA is trained only on publicly available data, and the 65B model is trained on 1.4 trillion tokens.

## 2.4 Efficient Implementation

LLaMA uses standard multi-head attention in every layer; it does not use grouped-query or multi-query attention.

An efficient causal multi-head attention implementation avoids storing the attention probabilities and skips the masked key/query scores.

LLaMA supports a context length of 2,048 tokens.
""",
    "dubey_2024_llama3": r"""
## 3.3 Scaling Laws and Over-Training

Under Chinchilla-style compute-optimal scaling, an 8B model would need only about 200B training tokens, yet Llama 3 8B is trained on roughly 15T tokens because over-training a small model keeps improving quality and makes inference cheaper.

## 3.4 Lineage of the Attention Design

Grouped-query attention entered the Llama family with the 34B and 70B Llama 2 models; LLaMA-1 used standard multi-head attention throughout.

## 3.5 Long-Context Pre-training

Llama 3 extends its context window from 8K to 128K tokens through continued pre-training on long documents in six gradual stages.

An attention mask prevents self-attention across different documents packed into the same training sequence.
""",
    "su_2021_rope": r"""
## 3.4 Properties of RoPE

Unlike additive absolute position embeddings, RoPE multiplies queries and keys by position-dependent rotations, so their inner product depends only on the relative distance m - n.

RoPE is applied only to the query and key vectors; the value vectors are not rotated.

## 3.5 Compatibility with Linear Attention

Because rotations preserve vector norms, RoPE can be combined with linear self-attention, unlike relative position biases that require the full attention matrix.
""",
    "peng_2023_yarn": r"""
## 2 Background: Position Interpolation

Position Interpolation linearly down-scales position indices to fit inside the original context window, but it compresses the high-frequency RoPE dimensions and hurts short-context performance.

YaRN improves on both Position Interpolation and NTK-aware interpolation by treating different RoPE frequency bands differently.

## 4 Deployment

YaRN changes only the RoPE frequencies and the attention temperature, so it requires no change to the model architecture and adds no inference overhead.

With dynamic scaling, YaRN can extend the context at inference time without any fine-tuning, although fine-tuning on a small amount of long data still gives better results.
""",
}

for _pid, _extra in PHASE5_APPENDIX.items():
    SYNTHETIC_SEED_MARKDOWN[_pid] = SYNTHETIC_SEED_MARKDOWN[_pid].rstrip("\n") + "\n" + _extra

_P5: Dict[str, List[Dict[str, Any]]] = {
    "vaswani_2017_transformer": [
        _rel("Multi-Head Attention", "ArchComponent", "CAUSES_BOTTLENECK", "Quadratic Self-Attention Complexity", "Bottleneck",
             "The complexity of self-attention per layer is $O(n^2 \\cdot d)$ in the sequence length n"),
        _rel("Transformer", "ModelFamily", "EVALUATED_ON", "WMT 2014 English-to-German", "BenchmarkOrDataset",
             "The big Transformer reaches 28.4 BLEU on the WMT 2014 English-to-German translation task", context="28.4 BLEU"),
        _rel("Transformer", "ModelFamily", "ADOPTS_COMPONENT", "Sinusoidal Positional Encoding", "ArchComponent",
             "The Transformer adopts sinusoidal positional encodings because the model contains no recurrence."),
    ],
    "ainslie_2023_gqa": [
        _rel("Grouped-Query Attention", "ArchComponent", "EXTENDS", "Multi-Query Attention", "ArchComponent",
             "We introduce grouped-query attention (GQA), a generalization of multi-query attention"),
        _rel("Grouped-Query Attention", "ArchComponent", "REQUIRES", "Uptraining", "TrainingMethod",
             "The converted checkpoint is then uptrained with only 5% of the original pre-training compute"),
    ],
    "deepseek_2024_v2": [
        _rel("Multi-head Latent Attention", "ArchComponent", "REQUIRES", "Decoupled RoPE", "ArchComponent",
             "MLA therefore carries positional information in an additional shared key k_t^R"),
        _rel("DeepSeek-V2", "ModelFamily", "ADOPTS_COMPONENT", "GRPO", "TrainingMethod",
             "DeepSeek-V2 Chat is aligned with supervised fine-tuning followed by reinforcement learning with GRPO."),
        _rel("DeepSeek-V2", "ModelFamily", "ADOPTS_COMPONENT", "Supervised Fine-Tuning", "TrainingMethod",
             "DeepSeek-V2 Chat is aligned with supervised fine-tuning followed by reinforcement learning with GRPO."),
    ],
    "deepseek_2024_v3": [
        _rel("DeepSeek-V3", "ModelFamily", "EXTENDS", "DeepSeek-V2", "ModelFamily",
             "DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture, both validated in DeepSeek-V2."),
        _rel("DeepSeek-V3", "ModelFamily", "ADOPTS_COMPONENT", "DeepSeekMoE", "ArchComponent",
             "DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and the DeepSeekMoE architecture"),
        _rel("DeepSeek-V3", "ModelFamily", "ADOPTS_COMPONENT", "Multi-Token Prediction (MTP)", "TrainingMethod",
             "DeepSeek-V3 adds a multi-token prediction (MTP) training objective"),
        _rel("DeepSeek-V3", "ModelFamily", "ADOPTS_COMPONENT", "FP8 Mixed Precision Training", "TrainingMethod",
             "We validate FP8 mixed precision training on an extremely large-scale model for the first time."),
        _rel("DeepSeek-V3", "ModelFamily", "CITES_INSPIRED_BY", "DeepSeek-R1", "ModelFamily",
             "reasoning capability is distilled from the DeepSeek-R1 series of models into DeepSeek-V3"),
    ],
    "deepseek_2025_r1": [
        _rel("DeepSeek-R1-Zero", "ModelFamily", "ADOPTS_COMPONENT", "GRPO", "TrainingMethod",
             "DeepSeek-R1-Zero is optimized with GRPO using rule-based accuracy and format rewards"),
        _rel("PPO", "TrainingMethod", "CAUSES_BOTTLENECK", "RLHF Critic Value Model Overhead", "Bottleneck",
             "PPO, by contrast, requires a separate value model (the critic) of similar size to the policy"),
        _rel("DeepSeek-R1", "ModelFamily", "ADOPTS_COMPONENT", "Cold-Start Supervised Fine-Tuning", "TrainingMethod",
             "DeepSeek-R1 first fine-tunes the base model on thousands of long chain-of-thought cold-start examples"),
        _rel("DeepSeek-R1", "ModelFamily", "REQUIRES", "DeepSeek-V3", "ModelFamily",
             "Both DeepSeek-R1-Zero and DeepSeek-R1 start from the DeepSeek-V3-Base checkpoint."),
        _rel("DeepSeek-R1-Zero", "ModelFamily", "REQUIRES", "DeepSeek-V3", "ModelFamily",
             "Both DeepSeek-R1-Zero and DeepSeek-R1 start from the DeepSeek-V3-Base checkpoint."),
    ],
    "rafailov_2023_dpo": [
        _rel("DPO", "TrainingMethod", "REPLACES_IN", "PPO", "TrainingMethod",
             "DPO replaces the PPO-based RLHF pipeline with a single supervised classification loss on preference pairs.",
             context="RLHF"),
        _rel("DPO", "TrainingMethod", "EVALUATED_ON", "TL;DR Summarization", "BenchmarkOrDataset",
             "On TL;DR summarization and Anthropic HH single-turn dialogue, DPO matches or exceeds PPO-based RLHF"),
    ],
    "kwon_2023_pagedattention": [
        _rel("vLLM", "InferenceTechnique", "REQUIRES", "PagedAttention", "ArchComponent",
             "vLLM is an LLM serving engine built on top of PagedAttention"),
    ],
    "hoffmann_2022_chinchilla": [
        _rel("Training Compute-Optimal Large Language Models (Chinchilla)", "Paper", "INTRODUCES", "Chinchilla", "ModelFamily",
             "Chinchilla uses 70B parameters trained on 1.4 trillion tokens"),
        _rel("Chinchilla", "ModelFamily", "OUTPERFORMS_ON", "MMLU", "BenchmarkOrDataset",
             "Chinchilla reaches an average accuracy of 67.6% on the MMLU benchmark, more than 7 points above Gopher.",
             context="67.6%, +7 over Gopher"),
    ],
    "touvron_2023_llama1": [
        _rel("LLaMA", "ModelFamily", "ADOPTS_COMPONENT", "Multi-Head Attention", "ArchComponent",
             "LLaMA uses standard multi-head attention in every layer"),
        _rel("Rotary Position Embedding", "ArchComponent", "REPLACES_IN", "Absolute Position Embedding", "ArchComponent",
             "LLaMA replaces absolute positional embeddings with rotary positional embeddings (RoPE)", context="LLaMA"),
    ],
    "dubey_2024_llama3": [
        _rel("Llama 2", "ModelFamily", "ADOPTS_COMPONENT", "Grouped-Query Attention", "ArchComponent",
             "Grouped-query attention entered the Llama family with the 34B and 70B Llama 2 models"),
        _rel("Llama 3", "ModelFamily", "ADOPTS_COMPONENT", "Rotary Position Embedding", "ArchComponent",
             "We increase the RoPE base frequency hyperparameter to 500,000"),
        _rel("Llama 3", "ModelFamily", "MODIFIES_HYPERPARAMETER", "RoPE Base Frequency", "HyperparameterOrMetric",
             "We increase the RoPE base frequency hyperparameter to 500,000", context="500,000"),
    ],
    "su_2021_rope": [
        _rel("Rotary Position Embedding", "ArchComponent", "REPLACES_IN", "Absolute Position Embedding", "ArchComponent",
             "Unlike additive absolute position embeddings, RoPE multiplies queries and keys by position-dependent rotations"),
    ],
    "peng_2023_yarn": [
        _rel("YaRN", "TrainingMethod", "EXTENDS", "Position Interpolation", "TrainingMethod",
             "YaRN improves on both Position Interpolation and NTK-aware interpolation"),
        _rel("YaRN", "TrainingMethod", "EXTENDS", "NTK-aware Interpolation", "TrainingMethod",
             "YaRN improves on both Position Interpolation and NTK-aware interpolation"),
        _rel("Position Interpolation", "TrainingMethod", "EXTENDS", "Rotary Position Embedding", "ArchComponent",
             "Position Interpolation linearly down-scales position indices to fit inside the original context window"),
    ],
}
for _pid, _rels in _P5.items():
    RECORDED_RELATIONSHIPS.setdefault(_pid, []).extend(_rels)

_CATEGORY_HINTS = {
    "attention": "attention", "rope": "positional", "position": "positional", "moe": "moe_routing",
}


def _entity(name: str, etype: str, hints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    ent: Dict[str, Any] = {"entity_type": etype, "canonical_name": name, "confidence": 0.93}
    if etype == "ArchComponent":
        low = name.lower()
        ent["category"] = next((cat for hint, cat in _CATEGORY_HINTS.items() if hint in low), "other")
    hint = (hints or {}).get(name.lower())
    if hint:  # full-corpus only: aliases / description / category recorded for this surface
        if hint.get("aliases"):
            ent["aliases"] = list(hint["aliases"])
        if hint.get("description"):
            ent["description"] = hint["description"]
        if hint.get("category") and etype == "ArchComponent":
            ent["category"] = hint["category"]
    return ent


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text)


class RecordedExtractionLLM:
    """Replays recorded triples whose evidence quote appears verbatim in the chunk input.

    Signature matches `GeminiGraphExtractor`'s `llm_callable`: (system_instruction, user_input,
    response_schema) -> str. `calls` counts invocations so tests can assert cache behavior.
    """

    def __init__(self, recordings: Optional[Dict[str, List[Dict[str, Any]]]] = None, full_corpus: bool = False) -> None:
        """`full_corpus=True` adds the 38-paper recordings and entity hints (data/fixtures/full_corpus)."""
        self.entity_hints: Dict[str, Dict[str, Any]] = {}
        if full_corpus:
            from src.ingestion.full_corpus_fixtures import full_corpus_recordings, load_full_corpus_fixtures

            recordings = full_corpus_recordings(RECORDED_RELATIONSHIPS if recordings is None else recordings)
            self.entity_hints = load_full_corpus_fixtures()["entity_hints"]
        self.recordings = RECORDED_RELATIONSHIPS if recordings is None else recordings
        self.calls = 0

    def __call__(self, system_instruction: str, user_input: str, response_schema: Dict[str, Any]) -> str:
        self.calls += 1
        text = _normalize_ws(user_input)
        relationships: List[Dict[str, Any]] = []
        entities: Dict[tuple, Dict[str, Any]] = {}
        for rels in self.recordings.values():
            for rel in rels:
                if _normalize_ws(rel["evidence_quote"]) in text:
                    relationships.append(rel)
                    for name, etype in ((rel["source_name"], rel["source_type"]), (rel["target_name"], rel["target_type"])):
                        entities.setdefault((name.lower(), etype), _entity(name, etype, self.entity_hints))
        return json.dumps({"entities": list(entities.values()), "relationships": relationships})