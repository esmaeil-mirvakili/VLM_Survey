# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** Learning Transferable Visual Models From Natural Language Supervision
- **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- **Venue / Year:** ICML 2021 (PMLR 139:8748-8763); arXiv preprint posted February 26, 2021
- **Link (PDF / arXiv):** https://arxiv.org/abs/2103.00020 ; https://proceedings.mlr.press/v139/radford21a.html
- **Code / Models (if any):** https://github.com/openai/CLIP ; model card: https://raw.githubusercontent.com/openai/CLIP/main/model-card.md

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - CLIP shows that training a simple dual-encoder with a contrastive image-text objective on 400M web pairs (WIT) yields strong zero-shot transfer across many vision tasks. It reaches 76.2% zero-shot ImageNet top-1 (matching original ResNet-50) and improves robustness under natural distribution shift (main paper Sec. 2-3, Fig. 7).

## Positioning and scope

- **Problem addressed / motivation:**
  - Standard vision pretraining relies on fixed label sets (e.g., ImageNet), limiting transfer to new tasks/concepts; the paper tests whether web-scale natural language supervision can provide broader, task-agnostic transfer (Sec. 1, Sec. 2.1-2.2).
- **What’s new vs prior work (1–3 bullets):**
  - Web-scale natural-language supervision for vision at 400M image-text pairs (WIT), much larger than earlier caption datasets (Sec. 2.2).
  - A streamlined contrastive objective (symmetric CE over image-text similarities) replacing autoregressive caption prediction for better efficiency (Sec. 2.3, Fig. 2-3).
  - Prompt-based zero-shot classification + prompt ensembling, with large gains over contextless labels (Sec. 3.1.4, Fig. 4).
- **Key assumptions (data, compute, setting):**
  - Access to massive internet image-text data and very large compute (up to 592 V100 GPUs; Sec. 2.5).
  - English-centric text supervision and BPE tokenizer; strongest claims focus on English-language prompts (Sec. 2.4; model card).
  - Zero-shot quality depends on class naming/prompt construction (Sec. 3.1.4).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: Modified ResNet family (ResNet-D + antialiased blur pooling + attention pooling) and ViT family variants (Sec. 2.4-2.5).
  - Text/LLM backbone: 63M-parameter 12-layer, 512-width Transformer, 8 heads, masked self-attention (Sec. 2.4).
  - Fusion / connector mechanism: Separate encoders projected to shared embedding space; cosine similarity with learned temperature; no cross-attention between modalities (Fig. 3).
  - Input representation (patches, regions, tokens, OCR, etc.): Random square image crops; lower-cased BPE text (49,152 vocab), max sequence length 76 with [SOS]/[EOS] (Sec. 2.4).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Retrieval-style contrastive pretraining; classification at inference via text-label matching.
  - Uses prompting/instruction format? Y/N: Y
  - Any tool use (OCR, detector, retrieval index)? Y/N: N

## Training objective(s)
- **Objective(s) used (list):**
  - Symmetric contrastive cross-entropy over image-to-text and text-to-image logits (InfoNCE/N-pair style) (Sec. 2.3, Fig. 3).
- **Loss details (high-level, no math needed):**
  - In each batch of N aligned pairs, the model increases similarity for true pairs and suppresses N^2-N mismatched pairs; temperature is learned and clipped for stability (Sec. 2.3, 2.5).
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - WIT: 400M web image-text pairs built from ~500k query terms, with approximate class balancing up to 20k pairs/query (Sec. 2.2 + footnote).
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - Main results are single-stage contrastive pretraining, then zero-shot prompting; one extra high-res epoch for ViT-L/14@336px (Sec. 2.5).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - Batch size 32,768; AdamW + cosine LR; random square crop only augmentation; learned tau initialized from 0.07 and clipped; mixed precision, gradient checkpointing, sharded similarity compute (Sec. 2.5).

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - WIT (WebImageText): 400M image-text pairs from public internet sources; includes filtered/crawled web data and prior sources like YFCC100M (Sec. 2.2; model card).
- **Finetuning / instruction data (if any):**
  - No task-specific finetuning for zero-shot results. Linear-probe evaluations train logistic regression on frozen features (Appendix A.3).
- **Evaluation benchmarks (list all used):**
  - Core 27-dataset suite (Appendix A/Table 2): Food101, CIFAR10, CIFAR100, Birdsnap, SUN397, Stanford Cars, FGVC Aircraft, Pascal VOC 2007, DTD, Oxford-IIIT Pets, Caltech101, Flowers102, MNIST, FER2013, STL-10, EuroSAT, RESISC45, GTSRB, KITTI Distance, Country211, PatchCamelyon, UCF101, Kinetics700, CLEVR Counts, Hateful Memes, Rendered SST2, ImageNet.
  - Also includes Yahoo/SUN early zero-shot comparison (Table 1) and robustness sets including ImageNetV2, ImageNet-A, ImageNet-R, ImageNet-Sketch, ObjectNet, ImageNet-Vid, YouTube-BB (Sec. 3.4 + appendices).
  - Paper reports aggregate scaling across 39 evals on 36 datasets (Fig. 9).
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - Accuracy/top-1/top-5, mean-per-class accuracy, 11-point mAP (VOC), ROC-AUC (Hateful Memes), mean(top-1, top-5) for Kinetics700; retrieval Recall@K on Flickr30k/MSCOCO in appendices.
- **Any ablations / diagnostic tests that matter:**
  - Prompt engineering + ensembling vs contextless labels (Fig. 4).
  - Zero-shot vs few-shot vs linear probe comparisons (Fig. 6, Fig. 8).
  - Data overlap analysis (35 datasets; overlap impact mostly limited) (Sec. 5, Fig. 17).
  - Compute scaling of zero-shot error (Fig. 9).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - 76.2% zero-shot ImageNet top-1 and 95% top-5, roughly matching original supervised ResNet-50 without ImageNet labels (Sec. 3.1.3, Table 1).
  - Robustness gap on natural distribution shifts reduced by up to 75% versus similarly accurate supervised ImageNet models (Sec. 3.4, Fig. 7).
- **Where it underperforms (tasks/conditions):**
  - Weak on several specialized/abstract tasks in zero-shot settings (e.g., EuroSAT, KITTI Distance, PatchCamelyon, CLEVRCounts) (Fig. 5).
  - OCR is mixed: strong on rendered text-style tasks, weaker on handwritten/street-number settings such as MNIST/SVHN (appendix analysis).
- **Generalization notes (OOD, compositionality, robustness):**
  - Stronger OOD robustness than standard supervised baselines across ImageNet shifts, but absolute performance on many tasks still trails top fully supervised systems (Sec. 3.4, Sec. 6).
  - Zero-shot quality is strongly prompt- and taxonomy-dependent (Sec. 3.1.4).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Broad task transfer from one pretrained model without task-specific training (OCR, geo-localization, action recognition, etc.) (Sec. 3, appendices).
  - High compute efficiency for transfer relative to alternative pretraining baselines and smooth scaling with compute (Fig. 2, Fig. 9).
  - Better natural-shift robustness than similarly accurate supervised ImageNet models (Fig. 7).
- **Failure cases / limitations (explicitly reported):**
  - Zero-shot CLIP is often only competitive with a ResNet-50 linear-probe baseline and remains below overall SOTA in many settings (Sec. 6).
  - Authors estimate roughly 1000x more compute may be needed for overall SOTA zero-shot performance (Sec. 6).
  - Few-shot transition can be counter-intuitive (performance can dip from zero-shot to few-shot) (Sec. 6).
  - Performance and bias are sensitive to class design; FairFace-related tests show meaningful disparities (Sec. 7 + appendix/model card).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Low in closed-set ranking, but semantic overreach can happen when prompts are underspecified or polysemous.
  - Weak grounding:
    - Moderate for tasks needing precise contextual disambiguation beyond label text.
  - OCR/text-in-image:
    - Stronger on rendered/clean text; weaker on blurry/handwritten/low-res text.
  - Counting:
    - Weak (explicitly poor on CLEVRCounts and related counting settings).
  - Spatial reasoning:
    - Likely weaker on spatially precise tasks not well represented in caption-like supervision.
  - Evaluation artifact / benchmark weakness:
    - Prompt/template choice and class-name mapping significantly move results; benchmark taxonomies can distort comparisons.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 3.1.3 / Tab. 1 / Pg. 5-6  
   - **Why it matters (1 line):** Shows that web-scale contrastive pretraining can match a classic supervised ImageNet baseline in zero-shot mode.

2. **Claim:**  
   - **Where:** Sec. 3.1.4 / Fig. 4 / Pg. 6-7  
   - **Why it matters (1 line):** Demonstrates that inference-time prompt design is a first-order performance lever, not a minor detail.

3. **Claim (optional):**  
   - **Where:** Sec. 3.4 / Fig. 7 / Pg. 10-11  
   - **Why it matters (1 line):** Indicates CLIP improves robustness under natural distribution shift, not just in-distribution benchmarks.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - CLIP is a dual-encoder VLM that couples a vision backbone (modified ResNet variants or ViTs) with a Transformer text encoder and trains both with a symmetric contrastive objective over image-text pairs. Instead of generating captions, the model learns a shared embedding space where matched pairs are close and mismatched pairs are far apart, with a learned temperature controlling similarity scale. The training regime is deliberately simple but very large-scale: 400M web image-text pairs (WIT), batch size 32,768, AdamW + cosine decay, and aggressive systems optimizations (mixed precision, gradient checkpointing, sharded similarity computation). At inference, CLIP performs classification by embedding textual class prompts and ranking similarities, enabling zero-shot transfer. Prompt formatting matters: replacing bare class names with task-aware templates and ensembling prompt variants yields substantial gains (about +5 ImageNet points in their analysis). Empirically, CLIP matches original ResNet-50 on zero-shot ImageNet and shows stronger robustness to natural distribution shifts, but it still underperforms on several specialized tasks (e.g., remote sensing, counting, some medical-like domains) and remains below overall SOTA on many benchmarks. The paper also stresses sensitivity to label taxonomy and bias risks, so evaluation design and prompt protocol are integral to interpreting reported scores.

## Table fields for the final comparison table
- **Model name + year:** CLIP (2021)
- **Family:** dual-encoder
- **Vision backbone:** Modified ResNet (RN50/101/x4/x16/x64) and ViT (B/32, B/16, L/14, L/14@336px)
- **Language backbone:** 12-layer 512-wide Transformer text encoder (63M params), masked self-attention, BPE
- **Fusion/connector:** Shared embedding space + cosine similarity with learned temperature (no cross-attention fusion)
- **Objective(s):** Symmetric image-text contrastive cross-entropy
- **Data type/scale (high-level):** 400M web image-text pairs (WIT)
- **Key evals:** 27-dataset transfer suite + ImageNet shift robustness suites + broader 36-dataset/39-eval analysis
- **Known failure modes (1–3 bullets):**
  - Prompt/class taxonomy sensitivity and polysemy-related errors
  - Weakness on counting and some specialized distribution shifts
  - Few-shot transition instability vs zero-shot behavior

## Notes / TODO
- **Open questions to verify later:**
  - Exact site-level composition of WIT is not released; reproducibility depends on substitute web-scale datasets.
  - How much of later CLIP-family progress comes from better data curation vs architecture/objective changes?
- **Follow-up papers this cites (add to reading list):**
  - ConVIRT (Zhang et al., 2020)
  - VirTex (Desai and Johnson, 2020)
  - Visual N-Grams (Li et al., 2017)
  - BiT (Kolesnikov et al., 2019), EfficientNet/Noisy Student (Tan and Le, 2019; Xie et al., 2020)
