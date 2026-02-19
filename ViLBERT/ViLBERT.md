# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks
- **Authors:** Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee
- **Venue / Year:** NeurIPS 2019 (Advances in Neural Information Processing Systems 32); arXiv v1 on August 6, 2019
- **Link (PDF / arXiv):** https://arxiv.org/abs/1908.02265 ; https://papers.nips.cc/paper/8297-vilbert-pretraining-task-agnostic-visiolinguistic-representations-for-vision-and-language-tasks
- **Code / Models (if any):** Official code + pretrained checkpoints: https://github.com/jiasenlu/vilbert_beta (deprecated) and https://github.com/facebookresearch/vilbert-multi-task (archived, includes ViLBERT references/checkpoints)

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - ViLBERT introduces a two-stream BERT-style architecture for vision-language pretraining, where visual and textual streams communicate via co-attentional transformer layers.
  - Pretraining on Conceptual Captions with masked multimodal modeling and image-text alignment improves transfer across VQA, VCR, referring expressions, and caption-based retrieval.

## Positioning and scope

- **Problem addressed / motivation:**
  - Prior V+L systems mostly learned grounding per task, limiting transfer. ViLBERT asks whether visual grounding itself can be pretrained once and reused broadly.
- **What’s new vs prior work (1–3 bullets):**
  - Two-stream design with cross-stream co-attention (instead of a single unified stream).
  - BERT-like multimodal pretraining objectives adapted to both text tokens and region features.
  - Strong transfer focus: one pretrained backbone with minor task heads for multiple heterogeneous V+L tasks.
- **Key assumptions (data, compute, setting):**
  - Uses detector-based region features (Faster R-CNN on Visual Genome), not end-to-end pixel training.
  - Relies on weakly aligned web alt-text captions (Conceptual Captions) at large scale.
  - Assumes BERT initialization and substantial compute for multimodal pretraining/fine-tuning.

## Model type and architecture
- **Architecture type (pick one):** `cross-attn fusion`
- **High-level design:**
  - Vision encoder: Region features from pretrained Faster R-CNN (ResNet-101), typically 10-36 boxes, plus spatial 5D box encoding and a special `IMG` token.
  - Text/LLM backbone: BERTBASE-initialized text stream (12-layer transformer, 12 heads, hidden size 768).
  - Fusion / connector mechanism: Co-attentional transformer layers exchange keys/values across streams (image-conditioned language attention and language-conditioned image attention).
  - Input representation (patches, regions, tokens, OCR, etc.): Detector region embeddings + WordPiece text tokens (`CLS`, `SEP`) with segment/position-style embeddings.
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: hybrid discriminative transfer (classification + retrieval ranking), not generative pretraining.
  - Uses prompting/instruction format? Y/N: N
  - Any tool use (OCR, detector, retrieval index)? Y/N: Y (external object detector; hard negatives in retrieval fine-tuning).

## Training objective(s)
- **Objective(s) used (list):**
  - Masked multi-modal modeling:
  - Masked language modeling (text tokens)
  - Masked region semantic prediction (predict detector class distribution via KL divergence)
  - Multi-modal alignment prediction (binary aligned vs not-aligned image-caption pair)
- **Loss details (high-level, no math needed):**
  - MLM-style reconstruction for masked text, KL loss for masked region semantics, and binary classification loss for alignment; pretraining losses are equally weighted.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy: pretraining on Conceptual Captions only (3.3M raw, ~3.1M usable after broken links); negatives for alignment are created by random image/caption replacement.
  - Multi-stage training? (pretrain → finetune → instruction tuning): pretrain on Conceptual Captions -> task-specific fine-tuning per benchmark.
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.): 15% masking on both text and regions; regions are zeroed 90% when masked; 8 TitanX GPUs, batch 512, 10 epochs, Adam 1e-4 with warmup+linear decay.

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - Conceptual Captions: ~3.3M weakly aligned web image-caption pairs; authors report training with ~3.1M valid pairs.
- **Finetuning / instruction data (if any):**
  - VQA 2.0, VCR, RefCOCO+, Flickr30k retrieval; plus zero-shot Flickr30k retrieval diagnostic using only pretrained alignment scoring.
- **Evaluation benchmarks (list all used):**
  - VQA 2.0
  - VCR (Q->A, QA->R, Q->AR)
  - RefCOCO+
  - Flickr30k caption-based image retrieval
  - Zero-shot Flickr30k retrieval
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - Accuracy-style scores (VQA, VCR, RefCOCO+) and Recall@1/5/10 for retrieval (standard and zero-shot).
- **Any ablations / diagnostic tests that matter:**
  - Two-stream vs single-stream architecture and no-pretraining baselines (Table 1).
  - Visual stream depth sweep (2/4/6/8 layers) showing task-dependent best depth (Table 2).
  - Pretraining data fraction sweep (0/25/50/100%) showing monotonic gains with more data (Table 3).
  - Qualitative zero-shot caption sampling from pretrained model to inspect learned grounding.

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - ViLBERT reports SOTA at publication time on all four core transfer tasks in their setup: VQA (70.55 test-dev), VCR (54.04 Q->AR val), RefCOCO+ (72.34 val), and Flickr30k retrieval (R@1 58.20).
  - Zero-shot retrieval after pretraining is non-trivial (R@1 31.86) despite no target-task fine-tuning.
- **Where it underperforms (tasks/conditions):**
  - Zero-shot retrieval is much lower than fine-tuned retrieval (31.86 vs 58.20 R@1).
  - Depth effects are not uniform: VCR/RefCOCO+ can prefer shallower variants while retrieval benefits from deeper visual streams.
  - Pretraining captions can be noisy/editorialized due to web alt-text collection, affecting direct generation quality.
- **Generalization notes (OOD, compositionality, robustness):**
  - Performance increases monotonically with pretraining data scale in their experiments (Table 3), suggesting scaling headroom.
  - Gains transfer across tasks with minimal task-specific head changes, indicating reusable grounded representations.
  - Strong zero-shot retrieval suggests some cross-dataset alignment generalization, though below fine-tuned performance.

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Clear transfer gains over both task-specific baselines and strong internal ablations.
  - Architecture is reusable and easy to adapt (typically classifier heads only).
  - Demonstrates learned visual-linguistic alignment before task fine-tuning.
- **Failure cases / limitations (explicitly reported):**
  - Conceptual Captions noise introduces non-visual/editorialized language artifacts in pretrained generations (Sec. 4 qualitative discussion).
  - Bidirectional BERT-style decoding for generation is acknowledged as an open issue; standard greedy/beam decoding is not directly applicable.
  - Scope exclusions noted by authors: long image-text sequences (dialog/video/embodied tasks) are not directly handled in this setup.
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination: Moderate risk when forced into generation; model was not pretrained as an autoregressive captioner.
  - Weak grounding: Reduced vs non-pretrained baselines, but still limited by detector proposal quality and weak web-text supervision.
  - OCR/text-in-image: Likely weak; no OCR-specific objective or text-in-image pipeline.
  - Counting: Region-based features can help coarse counting, but dense counting remains detector-limited.
  - Spatial reasoning: Better than older baselines but still proposal-level; fine-grained geometry can be brittle.
  - Evaluation artifact / benchmark weakness: Some benchmarks may reward priors; transfer claims partly depend on detector features and benchmark-specific setups.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 4 / Tab. 1 / Pg. 6-7  
   - **Why it matters (1 line):** It provides direct evidence that the pretrained two-stream model beats both task-specific baselines and internal non-pretrained counterparts.

2. **Claim:**  
   - **Where:** Sec. 4 / Tab. 3 / Pg. 7  
   - **Why it matters (1 line):** It supports a scaling trend: more Conceptual Captions pretraining data consistently improves downstream transfer.

3. **Claim (optional):**  
   - **Where:** Sec. 2.2 + Sec. 4 / Fig. 1 + Tab. 1 / Pg. 2 and Pg. 6-7  
   - **Why it matters (1 line):** It ties the architectural novelty (co-attention two-stream fusion) to empirical gains over a single-stream ablation.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - ViLBERT extends BERT to multimodal pretraining by introducing a two-stream transformer architecture: one stream processes detector-based image regions and the other processes text tokens, and co-attentional layers exchange cross-modal context between them. Pretraining is done on Conceptual Captions using two BERT-style proxy objectives adapted to vision-language data: masked multimodal modeling (mask text tokens and image regions, reconstructing token identities and detector-derived region semantics) and image-text alignment prediction (classify whether caption and image match). This design separates modality-specific processing while still enabling cross-modal fusion, and the paper shows it outperforms a single-stream alternative under similar conditions. In transfer, ViLBERT is adapted to VQA, VCR, RefCOCO+, and Flickr30k retrieval with relatively lightweight task heads and end-to-end fine-tuning. Reported results show broad gains over prior task-specific models and over non-pretrained variants, indicating that pretraining learns reusable grounding rather than task-bound heuristics. Key caveats are also explicit: pretraining captions are noisy web alt-text and can produce editorialized/non-visual language artifacts, and the bidirectional BERT-style setup is not naturally suited for standard autoregressive decoding. The model also inherits limits from detector-based region representations, which constrain very fine-grained spatial/detail reasoning.

## Table fields for the final comparison table
- **Model name + year:** ViLBERT (2019)
- **Family:** cross-attn fusion
- **Vision backbone:** Faster R-CNN (ResNet-101) region features + spatial box embeddings
- **Language backbone:** BERTBASE-initialized transformer text stream
- **Fusion/connector:** Two-stream co-attentional transformer layers (cross-stream key/value attention)
- **Objective(s):** Masked multimodal modeling + image-text alignment prediction
- **Data type/scale (high-level):** Conceptual Captions web image-alt-text pairs (~3.1M used)
- **Key evals:** VQA 2.0, VCR, RefCOCO+, Flickr30k retrieval (plus zero-shot retrieval)
- **Known failure modes (1–3 bullets):**
  - Quality/noise issues from weak web alt-text supervision
  - Dependence on detector proposals/features for visual grounding quality
  - Limited native text generation behavior from bidirectional pretraining

## Notes / TODO
- **Open questions to verify later:**
  - How much of ViLBERT’s gain remains with modern end-to-end ViT encoders instead of detector regions?
  - What is the best objective replacement for alignment prediction under larger modern multimodal corpora?
- **Follow-up papers this cites (add to reading list):**
  - BERT (Devlin et al., 2018)
  - Faster R-CNN (Ren et al., 2015)
  - Bottom-Up and Top-Down Attention (Anderson et al., 2018)
  - SCAN (Lee et al., 2018)
  - R2C / VCR benchmark paper (Zellers et al., 2019)
