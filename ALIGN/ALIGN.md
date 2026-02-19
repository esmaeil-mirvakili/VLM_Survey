# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
- **Authors:** Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig
- **Venue / Year:** ICML 2021 (PMLR 139)
- **Link (PDF / arXiv):** https://proceedings.mlr.press/v139/jia21b/jia21b.pdf ; https://arxiv.org/abs/2102.05918
- **Code / Models (if any):** No official training code or released pretrained checkpoints are provided in the paper/official pages.

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - ALIGN shows that a simple dual-encoder (EfficientNet + BERT) trained contrastively on 1.8B noisy web image-alt-text pairs can achieve SOTA image-text retrieval and strong transfer (e.g., 76.4% zero-shot ImageNet top-1, 88.64% fine-tuned ImageNet top-1). The core claim is that dataset scale can compensate for noise with minimal filtering.

## Positioning and scope

- **Problem addressed / motivation:**
  - Vision and vision-language pretraining often depends on expensive curated labels/captions; ALIGN asks whether minimally cleaned web alt-text at very large scale can replace heavy curation.
- **What’s new vs prior work (1–3 bullets):**
  - Uses a much larger but noisier 1.8B image-text corpus built by relaxing Conceptual Captions cleaning (Sec. 3).
  - Keeps architecture/objective simple: dual-encoder + normalized softmax contrastive loss (Sec. 4.1).
  - Beats prior cross-attention-heavy retrieval systems on Flickr30K/MSCOCO/CxC while preserving efficient retrieval inference (Sec. 5.1, Tables 1-3).
- **Key assumptions (data, compute, setting):**
  - Access to web-scale alt-text and willingness to tolerate noisy supervision.
  - Very large training compute (1024 TPUv3 cores; effective batch size 16,384; Sec. 5).
  - Main model is English-focused; multilingual extension requires separate multilingual data/vocab (Sec. 8).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: EfficientNet family; main reported model uses EfficientNet-L2. Global pooled features; classification-head 1x1 conv not trained during pretrain (Sec. 4.1, Sec. 5).
  - Text/LLM backbone: BERT family; main reported model uses BERT-Large with [CLS] representation (Sec. 4.1, Sec. 5).
  - Fusion / connector mechanism: Independent image/text towers mapped to a joint embedding space; cosine similarity scoring; no cross-attention fusion block (Sec. 4.1).
  - Input representation (patches, regions, tokens, OCR, etc.): Images resized/cropped to 289x289 input; text tokenized with WordPiece, max 64 tokens, based on alt-text up to 20 unigrams (Sec. 5).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Retrieval-style contrastive pretraining and embedding matching.
  - Uses prompting/instruction format? Y/N: Y (for zero-shot classification, prompt templates/ensembling from CLIP are used).
  - Any tool use (OCR, detector, retrieval index)? Y/N: N (no detector/OCR in the model; retrieval index shown as a downstream demo).

## Training objective(s)
- **Objective(s) used (list):**
  - Symmetric normalized softmax contrastive losses: image->text classification + text->image classification (Sec. 4.1, Eq. 1-2).
- **Loss details (high-level, no math needed):**
  - Positive pairs are matched image-text pairs; negatives are other in-batch pairs. Embeddings are L2-normalized, cosine-scored, and temperature-scaled; temperature is learned.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - Single large noisy corpus of 1.8B English image-alt-text pairs with minimal filtering (image quality/aspect filters, text frequency/length filters, near-duplicate test-image removal).
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - Pretrain once on noisy alt-text; then transfer via zero-shot, frozen-feature linear head, or task fine-tuning.
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - LAMB optimizer; label smoothing 0.1; LR warmup to 1e-3 in 10k steps then decay over 1.2M steps; batch 16,384 across 1024 TPUv3 cores; cross-core in-batch negatives; learned temperature.

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - 1.8B noisy English web image-alt-text pairs derived from Conceptual Captions-style pipeline with relaxed cleaning (Sec. 3).
- **Finetuning / instruction data (if any):**
  - Retrieval fine-tuning on Flickr30K and MSCOCO train splits (Sec. 5.1); classification fine-tuning on ImageNet and smaller fine-grained datasets (Sec. 5.3).
- **Evaluation benchmarks (list all used):**
  - Image-text retrieval: Flickr30K, MSCOCO.
  - Multimodal retrieval/similarity: CxC (i2t/t2i/t2t/i2i + STS/SIS/SITS).
  - Visual classification transfer: ImageNet, ImageNet-R, ImageNet-A, ImageNet-V2, VTAB-19, Flowers102, Oxford-IIIT Pets, Stanford Cars, Food101.
  - Multilingual retrieval: Multi30K (Sec. 8).
  - Supplementary diagnostic: SimLex-999 for language-side embedding behavior.
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - Recall@K (retrieval), top-1/top-5 accuracy (classification), mean recall mR (Multi30K), Spearman correlation (CxC similarity, SimLex).
- **Any ablations / diagnostic tests that matter:**
  - Backbone scaling and model-size interactions (Fig. 3).
  - Embedding dimension / in-batch negatives / temperature ablations (Table 8).
  - Dataset size-quality tradeoffs (Tables 9-10).
  - Prompt ensembling effect for zero-shot classification (+2.9% on ImageNet top-1; Sec. 5.2).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - Retrieval SOTA on Flickr30K/MSCOCO in both zero-shot and fine-tuned settings (e.g., Flickr30K fine-tuned R@1: 95.3 i2t / 84.9 t2i; Table 1).
  - Strong classification transfer: 76.4% zero-shot ImageNet top-1 and 88.64% fine-tuned ImageNet top-1 (Tables 4-5), plus VTAB 79.99 mean (Table 6).
- **Where it underperforms (tasks/conditions):**
  - On ImageNet-A zero-shot, ALIGN trails CLIP (75.8 vs 77.2, Table 4).
  - Intra-modal CxC tasks (text-text/image-image, STS/SIS) improve less than inter-modal retrieval; some metrics are slightly below prior methods (Sec. 5.1 discussion after Tables 2-3).
  - Supplementary language diagnostic shows weaker adjective/verb-related SimLex behavior than GloVe.
- **Generalization notes (OOD, compositionality, robustness):**
  - Shows robustness on distribution-shifted ImageNet variants (notably strong on ImageNet-R; Table 4).
  - Embeddings support compositional image+text queries and multilingual transfer (Sec. 7-8, Fig. 5, Table 11).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Scale-over-curation works: noisy data + simple objective still reaches SOTA retrieval and strong transfer.
  - Efficient dual-encoder outperforms heavier cross-attention systems on key retrieval benchmarks.
  - Clear scaling trends with larger image/text backbones and larger data (Sec. 6).
- **Failure cases / limitations (explicitly reported):**
  - Cross-modal objective is less effective for intra-modal similarity tasks (Sec. 5.1 discussion).
  - Authors explicitly flag potential harms: harmful text content, stereotype reinforcement, demographic skew, and misuse risks (Sec. 10).
  - Model/data require further analysis and balancing before practical deployment (Sec. 10).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Low in ranking tasks, but semantic overreach can appear when noisy alt-text associations are spurious.
  - Weak grounding:
    - Moderate risk in fine-grained attribute grounding because supervision is weak/noisy and alt-text may be off-image.
  - OCR/text-in-image:
    - Not a designed strength (no OCR pathway); performance depends on whether textual cues are captured implicitly.
  - Counting:
    - Likely weak for precise counting due global contrastive objective and coarse weak supervision.
  - Spatial reasoning:
    - Likely weaker than region/cross-attention models on detailed spatial relations.
  - Evaluation artifact / benchmark weakness:
    - Zero-shot scores depend on prompt templates/ensembling; paper reports a sizable +2.9% gain from this choice.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 3 + Sec. 4.1 / Fig. 2 / Pg. 2-3  
   - **Why it matters (1 line):** Supports the central thesis that minimal-cleaning, web-scale noisy supervision can still train useful vision-language representations.

2. **Claim:**  
   - **Where:** Sec. 5.1 / Tab. 1 / Pg. 4  
   - **Why it matters (1 line):** Shows large empirical gains over prior retrieval methods, including stronger cross-attention baselines.

3. **Claim (optional):**  
   - **Where:** Sec. 5.2 / Tab. 4 / Pg. 4-5  
   - **Why it matters (1 line):** Demonstrates practical zero-shot classification viability and sensitivity to prompt design (+2.9 ImageNet top-1 with ensembling).

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - ALIGN is a large-scale dual-encoder vision-language model that pairs an EfficientNet image tower with a BERT text tower and trains both from scratch using a symmetric contrastive normalized-softmax objective. Instead of relying on heavily curated image-caption corpora, ALIGN is pretrained on a 1.8B web image-alt-text dataset built with only lightweight frequency and quality filtering. During training, matched pairs are positives and all other in-batch pairs are negatives; embeddings are L2-normalized and temperature-scaled, with the temperature learned jointly. This setup is intentionally simple but compute-heavy (1024 TPUv3, effective batch size 16,384), and the paper argues that scale compensates for supervision noise. Empirically, ALIGN sets strong retrieval results on Flickr30K/MSCOCO/CxC and reaches 76.4% top-1 zero-shot on ImageNet while also delivering 88.64% top-1 after ImageNet fine-tuning. The paper also highlights two caveats important for evaluation: first, zero-shot performance is materially affected by prompt-template ensembling; second, although inter-modal retrieval is strong, intra-modal similarity gains are weaker, indicating objective-induced bias toward cross-modal matching. Finally, authors explicitly caution about social risks from web-alt-text supervision (harmful language, stereotype reinforcement, demographic skew, and misuse), making safety analysis a required part of deployment.

## Table fields for the final comparison table
- **Model name + year:** ALIGN (2021)
- **Family:** dual-encoder
- **Vision backbone:** EfficientNet (B1-B7, L2; main result uses EfficientNet-L2)
- **Language backbone:** BERT (Mini/Medium/Base/Large; main result uses BERT-Large)
- **Fusion/connector:** Shared embedding space with cosine similarity (no cross-attention fusion)
- **Objective(s):** Symmetric image-text contrastive normalized softmax (i2t + t2i)
- **Data type/scale (high-level):** 1.8B noisy web image-alt-text pairs (English) with minimal filtering
- **Key evals:** Flickr30K, MSCOCO, CxC, ImageNet(+R/A/V2), VTAB-19, Flowers/Pets/Cars/Food101, Multi30K
- **Known failure modes (1–3 bullets):**
  - Intra-modal semantic similarity weaker than inter-modal retrieval
  - Sensitivity to prompt-template choices in zero-shot classification
  - Risk of inherited web-data harms (bias, harmful text associations, misuse potential)

## Notes / TODO
- **Open questions to verify later:**
  - Reproducibility remains limited because the exact 1.8B training corpus is not publicly released.
  - Confirm whether any later official Google release includes reusable ALIGN checkpoints/training code.
- **Follow-up papers this cites (add to reading list):**
  - CLIP (Radford et al., 2021)
  - UNITER (Chen et al., 2020)
  - ImageBERT (Qi et al., 2020)
  - VSE++ (Faghri et al., 2018)
  - VirTex (Desai and Johnson, 2020)
