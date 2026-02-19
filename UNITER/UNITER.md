# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** UNITER: UNiversal Image-TExt Representation Learning
- **Authors:** Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu
- **Venue / Year:** ECCV 2020 (16th European Conference on Computer Vision, August 2020); arXiv version v3 July 17, 2020
- **Link (PDF / arXiv):** https://arxiv.org/abs/1909.11740 ; https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750103.pdf
- **Code / Models (if any):** Official repo with released UNITER-base / UNITER-large checkpoints: https://github.com/ChenRocks/UNITER

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - UNITER is a single-stream cross-modal Transformer pre-trained for generic image-text understanding via four objectives (MLM, MRM, ITM, WRA) over large mixed caption corpora.
  - Its two key training ideas are conditional masking and OT-based word-region alignment, and it reports SOTA across a broad suite of V+L tasks at publication time.

## Positioning and scope

- **Problem addressed / motivation:**
  - Prior V+L systems were often task-specific and hard to transfer. UNITER targets a universal joint representation that can be fine-tuned across heterogeneous V+L tasks.
- **What’s new vs prior work (1–3 bullets):**
  - Conditional masking: mask one modality while keeping the other fully visible to reduce cross-modal misalignment.
  - Word-Region Alignment (WRA) objective using Optimal Transport for explicit fine-grained alignment, beyond global ITM matching.
  - Strong ablation-driven task mix (MLM+ITM+MRC-kl+MRFR+WRA) and broad multi-task validation.
- **Key assumptions (data, compute, setting):**
  - Uses pre-extracted object-region features from a Faster R-CNN detector (Bottom-Up style), not end-to-end pixels.
  - Relies on large paired image-text pretraining corpora and careful overlap filtering versus downstream eval splits.
  - Compute is non-trivial (reported pretraining cost: ~882 V100 GPU-hours for base and ~3645 for large).

## Model type and architecture
- **Architecture type (pick one):** `cross-attn fusion`
- **High-level design:**
  - Vision encoder: Faster R-CNN pooled ROI region features + 7D geometric box features, projected into model hidden space.
  - Text/LLM backbone: BERT-style WordPiece token embeddings with position embeddings, then a multi-layer Transformer encoder.
  - Fusion / connector mechanism: single-stream self-attention over concatenated region and token embeddings (joint contextualization).
  - Input representation (patches, regions, tokens, OCR, etc.): object regions (detector features), bounding-box geometry, and subword tokens.
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: hybrid discriminative transfer (classification + ranking retrieval), not generative decoding.
  - Uses prompting/instruction format? Y/N: N
  - Any tool use (OCR, detector, retrieval index)? Y/N: Y (external object detector features; hard negatives for retrieval fine-tuning).

## Training objective(s)
- **Objective(s) used (list):**
  - MLM (Masked Language Modeling) conditioned on full image.
  - MRM (Masked Region Modeling) conditioned on full text:
  - MRC (region classification)
  - MRFR (feature regression)
  - MRC-kl (KL distillation from detector soft labels)
  - ITM (Image-Text Matching; positive vs negative pairs)
  - WRA (Word-Region Alignment via Optimal Transport distance)
- **Loss details (high-level, no math needed):**
  - Token prediction loss for MLM, regression/classification losses for MRM variants, binary matching loss for ITM, and OT-based alignment loss for WRA.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy: pretrain on in-domain (COCO+VG) and out-of-domain (CC+SBU) caption pairs; best downstream performance comes from combining both.
  - Multi-stage training? (pretrain → finetune → instruction tuning): pretrain then task-specific fine-tuning; plus optional second-stage pretraining on VCR data for stronger VCR transfer.
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.): conditional masking (vs joint random masking), OT WRA, strict overlap filtering against eval images, hard negatives in retrieval fine-tuning.

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - Four corpora: COCO captions, Visual Genome dense captions, Conceptual Captions, SBU captions.
  - Train split sizes (image-text pairs): COCO 533K, VG 5.06M, CC 3.0M, SBU 990K (~9.58M total).
  - Internal val split pairs: COCO 25K, VG 106K, CC 14K, SBU 10K.
- **Finetuning / instruction data (if any):**
  - Fine-tuned on VQA, VCR, NLVR2, SNLI-VE, Flickr30K/COCO retrieval, and RefCOCO/RefCOCO+/RefCOCOg.
- **Evaluation benchmarks (list all used):**
  - VQA v2
  - VCR (Q->A, QA->R, Q->AR)
  - NLVR2
  - SNLI-VE
  - Image-Text Retrieval on Flickr30K and COCO (including zero-shot Flickr setting)
  - Referring Expression Comprehension: RefCOCO / RefCOCO+ / RefCOCOg
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - VQA score (test-dev/test-std), classification accuracy, Recall@1/5/10 for retrieval, and Meta-Sum for ablation aggregation.
- **Any ablations / diagnostic tests that matter:**
  - Full pretraining-task ablation (Table 2): best mix is MLM+ITM+MRC-kl+MRFR+WRA.
  - Conditional masking vs joint random masking: conditional masking improves both pretraining curves and downstream aggregate score.
  - Dataset ablation: in-domain only > out-of-domain only; combining both gives best aggregate.
  - WRA ablations: largest gains on region-sensitive tasks (especially VQA and RefCOCO+).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - UNITER-large reaches strong best-in-table results across all listed tasks at publication time, e.g., VQA test-std 74.02, NLVR2 test-P 79.98, VCR Q->AR 62.80, SNLI-VE test 79.38.
  - Zero-shot Flickr retrieval is notably strong (R@1: IR 68.74, TR 83.60), and fine-tuned retrieval also improves over prior baselines.
- **Where it underperforms (tasks/conditions):**
  - UNITER-base is not the top model on every VQA comparison line (paper notes LXMERT uses extra downstream supervision that may help VQA).
  - Out-of-domain-only pretraining underperforms in-domain-only pretraining despite more images.
  - WRA helps less on more global-reasoning-heavy tasks (e.g., Flickr retrieval / NLVR2) than on fine-grained region reasoning tasks.
- **Generalization notes (OOD, compositionality, robustness):**
  - Adding out-of-domain data on top of in-domain data gives further gains, suggesting useful transfer from broader web captions.
  - A two-stage pretraining setup on task-specific distribution (VCR) yields substantial extra gains, indicating adaptable transfer pipelines.
  - Single-stream architecture remains highly competitive despite fewer parameters than some two-stream alternatives.

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Strong multi-task transfer across six V+L tasks with one shared pretraining recipe.
  - Clear ablation evidence for conditional masking and OT-based WRA.
  - Parameter-efficient relative to some contemporaries while still achieving SOTA.
- **Failure cases / limitations (explicitly reported):**
  - NLVR2 uses paired images, which mismatches pretraining input format (single image-text pair), requiring architectural adaptation during fine-tuning.
  - For RE comprehension, the commonly used detector features are acknowledged as potentially contaminated by COCO overlap; authors keep this for fair comparison and defer strict-clean setup to future work.
  - No dedicated ethics/bias section in the main paper; risk analysis is limited.
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination: Lower than decoder-only generators (model is discriminative), but still possible in retrieval/ranking errors.
  - Weak grounding: Improved by WRA, but may still struggle when detector misses key objects/attributes.
  - OCR/text-in-image: Likely weak relative to later OCR-integrated VLMs; UNITER has no dedicated OCR objective.
  - Counting: Dependent on region proposals and detector quality; likely brittle in dense scenes.
  - Spatial reasoning: Better than many peers at the time, but still constrained by region feature granularity and proposal coverage.
  - Evaluation artifact / benchmark weakness: Retrieval/VQA metrics can under-represent semantic equivalence; detector contamination caveat exists for some RE evaluations.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 4.2 / Tab. 2 / Pg. 9-10  
   - **Why it matters (1 line):** Shows the empirically best pretraining objective mix and quantifies gains from conditional masking and WRA.

2. **Claim:**  
   - **Where:** Sec. 4.3 / Tab. 3 / Pg. 11-12  
   - **Why it matters (1 line):** Documents broad SOTA-level transfer across VQA, VCR, NLVR2, SNLI-VE, retrieval, and referring expression tasks.

3. **Claim (optional):**  
   - **Where:** Sec. 4.2 / Tab. 2 (L11 vs L13 vs L14) / Pg. 9-10  
   - **Why it matters (1 line):** Demonstrates that combined in-domain + out-of-domain pretraining outperforms either source alone.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - UNITER is a single-stream multimodal Transformer that fuses detector-based region features and WordPiece text tokens in one shared self-attention stack, rather than using separate visual/text streams. The model is pre-trained on four image-text corpora (COCO, Visual Genome, Conceptual Captions, SBU) with four complementary objectives: MLM, MRM, ITM, and WRA. Two training design choices are central to its gains. First, UNITER uses conditional masking, masking one modality at a time while keeping the other intact, which reduces alignment ambiguity versus joint random masking. Second, it adds an OT-based word-region alignment objective to explicitly encourage fine-grained token-region correspondence beyond global matching. Ablations show the best recipe is MLM+ITM+MRC-kl+MRFR+WRA, and that combining in-domain and out-of-domain pretraining data gives the strongest aggregate transfer. UNITER-large reports leading results across diverse benchmarks at publication time, including VQA, VCR, NLVR2, SNLI-VE, retrieval, and referring expression comprehension. Reported caveats include dataset/task mismatch issues (e.g., NLVR2’s two-image inputs require architectural adjustment) and evaluation contamination concerns for referring-expression setups that reuse standard detector features trained on overlapping COCO images.

## Table fields for the final comparison table
- **Model name + year:** UNITER (2020)
- **Family:** cross-attn fusion
- **Vision backbone:** Faster R-CNN region features (Bottom-Up style) + box geometry embeddings
- **Language backbone:** BERT-style Transformer encoder (UNITER-base 12L, UNITER-large 24L)
- **Fusion/connector:** single-stream joint self-attention over concatenated region+token embeddings
- **Objective(s):** MLM, ITM, MRM (MRC/MRFR/MRC-kl), OT-based WRA
- **Data type/scale (high-level):** ~9.58M pretraining image-text pairs from COCO+VG+CC+SBU after cleaning
- **Key evals:** VQA, VCR, NLVR2, SNLI-VE, Flickr30K/COCO retrieval, RefCOCO/+/g
- **Known failure modes (1–3 bullets):**
  - Dependence on external detector quality and proposal coverage
  - Limited native handling of paired-image inputs (NLVR2 workaround needed)
  - Potential evaluation contamination issues in RE settings using standard detector features

## Notes / TODO
- **Open questions to verify later:**
  - How much of UNITER’s gains persist when replacing detector features with end-to-end ViT patch encoders?
  - What is the cleanest apples-to-apples RE benchmark protocol with strictly non-overlapping detector training data?
- **Follow-up papers this cites (add to reading list):**
  - ViLBERT (Lu et al., 2019)
  - LXMERT (Tan and Bansal, 2019)
  - VisualBERT (Li et al., 2019)
  - VL-BERT (Su et al., 2019)
  - Unicoder-VL (Li et al., 2019)
  - B2T2 (Alberti et al., 2019)
