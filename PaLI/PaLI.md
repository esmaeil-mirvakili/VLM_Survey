# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** PaLI: A Jointly-Scaled Multilingual Language-Image Model
- **Authors:** Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, et al. (Google Research)
- **Venue / Year:** ICLR 2023 (conference version); arXiv v4 dated June 5, 2023
- **Link (PDF / arXiv):** https://arxiv.org/abs/2209.06794
- **Code / Models (if any):** No public PaLI checkpoint release in the paper/model card (Appendix F says current version is not publicly available). Related infrastructure/components: `big_vision` and mT5/ViT backbones (partial PaLI support): https://github.com/google-research/big_vision

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - PaLI is a unified multilingual vision-language seq2seq model (`image + text -> text`) that scales both language and vision backbones jointly, instead of mostly scaling only language.
  - With a new multilingual web-scale dataset (WebLI) and multi-objective pretraining, PaLI-17B sets then-SOTA results across captioning and VQA while retaining strong language-only performance.

## Positioning and scope

- **Problem addressed / motivation:**
  - Prior large VLMs often had imbalanced scaling (very large LM, smaller vision tower) and were mostly English-centric. PaLI addresses multilingual, multi-task VLM learning with a single generation interface and balanced scaling.
- **What’s new vs prior work (1–3 bullets):**
  - Joint scaling study across both modalities, including a new ~4B ViT-e vision backbone paired with mT5-XXL.
  - WebLI: multilingual web-scale image-text corpus (109 languages) with OCR augmentation.
  - One promptable seq2seq interface for captioning, VQA, OCR-heavy tasks, multilingual transfer, and language-only retention checks.
- **Key assumptions (data, compute, setting):**
  - Assumes access to very large-scale noisy web data and heavy TPU compute (PaLI-17B pretraining reported on 1,024 TPUv4s for ~7 days).
  - Relies on prompt templates and text generation as a universal interface (open-vocabulary evaluation).
  - Uses OCR strings from automatic OCR service for OCR-sensitive tasks during training/fine-tuning in several setups.

## Model type and architecture
- **Architecture type (pick one):** `cross-attn fusion`
- **High-level design:**
  - Vision encoder: ViT-G (1.8B) or ViT-e (~3.9/4B), patch-token outputs.
  - Text/LLM backbone: mT5-Large (~1B) or mT5-XXL (~13B), encoder-decoder.
  - Fusion / connector mechanism: Vision patch tokens are fed into the encoder sequence; decoder generates text autoregressively via standard encoder-decoder attention.
  - Input representation (patches, regions, tokens, OCR, etc.): image patches + text tokens + task prompts; optional OCR text strings used for OCR-centric tasks.
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: generation
  - Uses prompting/instruction format? Y/N: Y
  - Any tool use (OCR, detector, retrieval index)? Y/N: Y (OCR service text; object-aware/object-detection style objectives in pretraining mix)

## Training objective(s)
- **Objective(s) used (list):**
  - Span corruption on text-only data
  - Split-captioning on WebLI alt-text
  - Captioning on CC3M-35L
  - OCR text generation on WebLI OCR data
  - English/cross-lingual VQA
  - English/cross-lingual VQG
  - Object-aware QA
  - Generative object detection
- **Loss details (high-level, no math needed):**
  - Teacher-forced seq2seq training with standard token-level softmax cross-entropy over generated text.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy: 8-task mixture totaling ~1.566B/1.6B examples (Table 9), with empirically chosen coefficients.
  - Multi-stage training? (pretrain → finetune → instruction tuning): Yes: large multi-task pretraining (224x224, one epoch) -> task fine-tuning; for PaLI-17B an extra high-res pretraining phase (588x588, 10k steps, 10M examples).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.): WebLI filtered to top 10% quality (~1B examples from 10B image pool); near de-duplication against many benchmarks; vision tower frozen in main pretraining then unfrozen in high-res continuation.

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - WebLI: ~10B images, ~12B alt-texts across 109 languages; plus ~29B OCR pairs extracted automatically. PaLI training uses filtered high-quality subset + mixture components for ~1.6B example coverage.
  - Additional mixture sources include CC3M-35L, VQ2A-CC3M-35L, Open Images/Visual Genome/Object365 for object-related tasks, and text-only corpus samples.
- **Finetuning / instruction data (if any):**
  - Task-specific fine-tuning on captioning and VQA benchmarks (e.g., COCO, VQAv2, TextVQA, OKVQA, multilingual translated splits such as COCO-35L, VQAv2-13L).
- **Evaluation benchmarks (list all used):**
  - Captioning: COCO, NoCaps, TextCaps, VizWiz-Cap, Crossmodal-3600 (via COCO-35L fine-tune).
  - VQA: VQAv2, OKVQA, TextVQA, VizWiz-QA, ST-VQA, xGQA, MaXM.
  - Language-only: SuperGLUE, XTREME tasks (XNLI, XQuAD, TyDiQA-GoldP).
  - Zero-shot classification: ImageNet, ImageNet-R, ImageNet-A, ImageNet-Sketch, ImageNet-v2, ObjectNet.
  - Additional appendix diagnostics include VQG and counting-related TallyQA.
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - CIDEr (captioning), VQA exact-match style accuracy, ANLS (ST-VQA), top-1/top-5 accuracy (ImageNet-style), F1/EM (XQuAD, TyDiQA), SuperGLUE aggregate score.
- **Any ablations / diagnostic tests that matter:**
  - Objective-mixture ablations (Table 6), including OCR/VQA/VQG/object-related removals.
  - Joint scaling analysis across language and vision capacity (Figure 2, Table 12).
  - Frozen vs fine-tuned vision tower effects (Table 15), multilingual data ablations (Table 16), with/without OCR input (Appendix C.5/Table 20).
  - Language retention vs mT5-XXL on SuperGLUE/XTREME (Table 11).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - PaLI-17B reports 149.1 CIDEr on COCO Karpathy test and 84.3 on VQAv2 test-dev/test-std in open-vocabulary generation.
  - Strong multilingual gains (e.g., Crossmodal-3600 and xGQA/MaXM improvements) and evidence that scaling vision (ViT-G -> ViT-e) materially improves multi-task VLM performance.
- **Where it underperforms (tasks/conditions):**
  - On NoCaps transfer, authors note COCO->NoCaps transfer is somewhat sub-optimal versus some English-only pretraining regimes.
  - OCR objective helps OCR-heavy tasks but slightly hurts captioning in ablations.
  - Low-resource languages remain substantially harder than high-resource languages in multilingual evaluations.
- **Generalization notes (OOD, compositionality, robustness):**
  - Positive OOD signs: NoCaps long-tail object description and ImageNet-OOD style zero-shot tests.
  - Language capability retention remains close to mT5-XXL despite multimodal-heavy training.
  - Multilingual pretraining generally helps both cross-lingual tasks and, in some settings, even English COCO CIDEr.

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Unified promptable seq2seq interface works across many tasks and languages without task-specific heads.
  - Joint scaling results show clear gains from increasing both vision and language capacity.
  - Strong OCR-aware and multilingual performance with broad benchmark coverage.
- **Failure cases / limitations (explicitly reported):**
  - May not describe very complex scenes thoroughly due to annotation limitations in source data (Appendix E).
  - Multilingual capabilities can degrade when fine-tuned only on English data (Appendix E).
  - Open-vocabulary evaluation may undercount correct paraphrases/synonyms as wrong; benchmark/evaluation bias concerns remain (Appendix E, ethics section).
  - Web-scale data may propagate undesirable/bias-prone content; quality varies by language coverage (ethics statement).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination: Moderate risk on open-ended generation, especially under long-tail multilingual prompts.
  - Weak grounding: Better than many peers on OCR/VQA, but complex multi-object scenes remain a weak point per authors.
  - OCR/text-in-image: Strong when OCR strings are provided; performance drops without OCR input on some tasks.
  - Counting: Improved with scale, but likely brittle on dense/count-heavy scenes (appendix counting diagnostics suggest room to improve).
  - Spatial reasoning: Object-aware/detection objectives help, but no dedicated strong spatial reasoning benchmark leadership claim.
  - Evaluation artifact / benchmark weakness: Exact-match generative scoring penalizes semantically correct paraphrases; cross-cultural benchmark bias still unresolved.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 4.5 / Fig. 2 / Pg. 8-9  
   - **Why it matters (1 line):** It supports the core thesis that scaling the vision tower (not only language) yields meaningful average gains across captioning and VQA tasks.

2. **Claim:**  
   - **Where:** Sec. 4.1 / Tab. 1 / Pg. 5-6  
   - **Why it matters (1 line):** It establishes a strong empirical result (149.1 CIDEr on COCO) while using standard cross-entropy fine-tuning rather than CIDEr optimization.

3. **Claim (optional):**  
   - **Where:** Sec. 3.2 + Appx A.2 / Tab. 9 / Pg. 4-5 and 19-20  
   - **Why it matters (1 line):** The multilingual, multi-objective pretraining mixture (including OCR/object-aware tasks) is presented as a direct driver of transfer quality across tasks/languages.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - PaLI is a multilingual encoder-decoder VLM that treats nearly all tasks as text generation conditioned on image and optional text prompts. Architecturally, it combines a large ViT image encoder (ViT-G or ViT-e) with mT5 language backbones, and scales both sides jointly rather than over-investing only in language parameters. Pretraining uses a broad 8-objective mixture that includes text-only span corruption, split-captioning, multilingual captioning, OCR text generation, VQA/VQG, and object-aware objectives, optimized with standard teacher-forced cross-entropy. Data scale is central: the WebLI corpus contributes roughly 10B images and multilingual alt-text across 109 languages, plus OCR-derived supervision; a filtered high-quality subset and additional task datasets form a ~1.6B-example training mixture. Empirically, PaLI-17B reports strong cross-benchmark results, including COCO captioning and open-vocabulary VQA gains, while preserving much of mT5-XXL’s language-only capability. The paper also identifies failure modes: complex-scene descriptions remain incomplete in some cases, multilingual ability can regress under English-only fine-tuning, and open-vocabulary exact-match evaluation can under-credit semantically correct outputs. These caveats make PaLI strong but still sensitive to data coverage, evaluation protocol, and downstream adaptation strategy.

## Table fields for the final comparison table
- **Model name + year:** PaLI (2023)
- **Family:** cross-attn fusion
- **Vision backbone:** ViT-G (1.8B) / ViT-e (~3.9-4B)
- **Language backbone:** mT5-Large (~1B) / mT5-XXL (~13B)
- **Fusion/connector:** Vision patch tokens integrated in encoder input of mT5-style encoder-decoder; autoregressive text decoder
- **Objective(s):** Multi-task generative CE: span corruption, split-cap, cap, OCR, VQA, VQG, object-aware QA, generative detection
- **Data type/scale (high-level):** Web-scale multilingual image-text + OCR; WebLI (10B images, 109 languages), training mixture ~1.6B examples
- **Key evals:** COCO/NoCaps/TextCaps/VizWiz-Cap, VQAv2/OKVQA/TextVQA/VizWiz-QA/ST-VQA, xGQA/MaXM, SuperGLUE+XTREME, ImageNet(+OOD)
- **Known failure modes (1–3 bullets):**
  - Complex scene under-description (explicitly noted)
  - Multilingual degradation after English-only fine-tuning
  - Open-vocabulary exact-match scoring mismatch and benchmark bias limitations

## Notes / TODO
- **Open questions to verify later:**
  - Was any full PaLI checkpoint later released outside this paper lineage, or are only descendant models (PaLI-X/PaLM-E variants) publicly accessible?
  - Which objective coefficients in Table 9 are most compute-efficient for smaller (<5B) replications?
- **Follow-up papers this cites (add to reading list):**
  - Flamingo (Alayrac et al., 2022)
  - CoCa (Yu et al., 2022)
  - BEiT-3 (Wang et al., 2022)
  - SimVLM (Wang et al., 2021)
  - OFA (Wang et al., 2022)
  - Unified-IO (Lu et al., 2022)
