# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** Flamingo: a Visual Language Model for Few-Shot Learning
- **Authors:** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan
- **Venue / Year:** NeurIPS 2022 (Conference paper); arXiv v2 dated November 15, 2022
- **Link (PDF / arXiv):** https://arxiv.org/abs/2204.14198 ; https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html
- **Code / Models (if any):** No official DeepMind training repo/checkpoints are linked on the paper pages. Community reproduction: https://github.com/mlfoundations/open_flamingo (explicitly marked as unaffiliated with DeepMind).

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - Flamingo proposes a connector-style VLM that injects visual information into a frozen large LM via a Perceiver Resampler and interleaved gated cross-attention layers. With in-context prompting only (no task-specific gradient updates), Flamingo-80B sets strong few-shot results across a broad multimodal benchmark suite and can beat several fine-tuned task specialists.

## Positioning and scope

- **Problem addressed / motivation:**
  - Existing contrastive vision-language models transfer well to classification-like tasks but are weak for open-ended generation (captioning, VQA, dialogue). Flamingo targets unified few-shot adaptation across image and video tasks via prompt-based text generation.
- **What’s new vs prior work (1–3 bullets):**
  - Supports arbitrarily interleaved image/video and text context in a single autoregressive model (Sec. 2, Fig. 3).
  - Bridges frozen vision and language backbones with a Perceiver Resampler plus interleaved GATED XATTN-DENSE blocks (Sec. 2.1-2.2, Fig. 4).
  - Trains on a mixed web corpus combining interleaved webpages (M3W) and paired image/video-text data (ALIGN, LTIP, VTP), which is shown by ablations to be critical (Sec. 2.4, Sec. 3.3, Table 3).
- **Key assumptions (data, compute, setting):**
  - Access to very large web-scale multimodal data and large TPU training budgets (Appendix B.1.2).
  - Dependence on pretrained frozen LMs implies inheriting LM priors, including hallucinations and social-bias risks (Sec. 5).
  - Few-shot quality depends on demonstration/prompt choices and does not scale cleanly to very high shot counts (Sec. 5, Appendix B.1.5).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: Frozen NFNet-F6, contrastively pretrained on ALIGN+LTIP (Sec. 2.1, Appendix B.1.3).
  - Text/LLM backbone: Frozen autoregressive LM; largest model uses frozen Chinchilla 70B, plus smaller 1.4B and 7B variants for Flamingo-3B/9B (Sec. 2.2, Appendix B.1.1).
  - Fusion / connector mechanism: Perceiver Resampler converts variable visual features into fixed 64 visual tokens, then GATED XATTN-DENSE layers are interleaved inside the frozen LM (Sec. 2.1-2.2, Fig. 4).
  - Input representation (patches, regions, tokens, OCR, etc.): Interleaved text and visual markers (`<image>`, `<EOC>`). M3W training samples use 256-token chunks with up to 5 images; videos are sampled at 1 FPS during training (Sec. 2.4, Appendix B.1.2).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Hybrid; autoregressive text generation for open-ended tasks and log-likelihood scoring for close-ended tasks.
  - Uses prompting/instruction format? Y/N: Y
  - Any tool use (OCR, detector, retrieval index)? Y/N: N (no external OCR/detector/retrieval tools in the model loop).

## Training objective(s)
- **Objective(s) used (list):**
  - Weighted autoregressive next-token negative log-likelihood over multiple datasets (Sec. 2.4, Eq. 2).
- **Loss details (high-level, no math needed):**
  - Model predicts each text token conditioned on prior text and preceding visual inputs in the interleaved sequence; per-dataset losses are weighted and summed.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - M3W (interleaved web pages, ~43M pages), ALIGN (1.8B image-alt-text pairs), LTIP (312M long image-text pairs), VTP (27M video-text pairs), with learned/selected dataset weights.
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - Main contribution is pretraining + in-context adaptation without task-specific updates; optional supervised fine-tuning is reported separately (Sec. 3.2).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - AdamW, warmup 5k steps to 1e-4 then constant LR, 500k steps, gradient accumulation over datasets (better than round-robin), tanh-gated cross-attn initialized at zero for stability, dataset weights 1.0/0.2/0.2/0.03 for M3W/ALIGN/LTIP/VTP (Appendix B.1.2).

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - Web-scale multimodal mix: M3W (~43M webpages), ALIGN (1.8B image-text pairs), LTIP (312M image-text pairs), VTP (27M short videos with text), plus dedup against many eval sets for paired datasets (Sec. 2.4, Appendix A.3.3-A.3.4).
- **Finetuning / instruction data (if any):**
  - Yes, fine-tuning experiments unfreeze additional components and use full task supervision on benchmarks where few-shot was not SOTA (Sec. 3.2, Table 2, Appendix B.2.2).
- **Evaluation benchmarks (list all used):**
  - 16 multimodal benchmarks in the main evaluation suite: COCO, VQAv2, OKVQA, Flickr30k, VizWiz, TextVQA, VisDial, HatefulMemes, VATEX, MSVDQA, YouCook2, MSRVTTQA, iVQA, RareAct, NextQA, STAR.
  - Additional classification/retrieval analyses in appendix: ImageNet-1k, Kinetics700, COCO/Flickr retrieval (Appendix B.2, Table 7-9).
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - CIDEr, VQA accuracy, top-1 accuracy, NDCG, ROC-AUC, iVQA accuracy, WUPS, mWAP, and retrieval recall metrics depending on task.
- **Any ablations / diagnostic tests that matter:**
  - Data-mixture ablations (removing M3W / video-text / image-text; replacing image-text with LAION) (Table 3).
  - Architecture ablations: gating, cross-attn variant/frequency, resampler choice, vision encoder strength, LM freezing (Table 3).
  - Scaling study over model size and number of shots (Fig. 2 and Sec. 3.1).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - Flamingo-80B posts strong 32-shot performance across the 16-task suite; examples include VQAv2 test-dev 67.6 and COCO CIDEr 113.8 without task-specific weight updates (Table 1/2).
  - With full fine-tuning, Flamingo reports new SOTA on five additional tasks, including VQAv2 test-dev 82.0 and COCO CIDEr 138.1 (Sec. 3.2, Table 2).
- **Where it underperforms (tasks/conditions):**
  - Authors explicitly note classification lags top contrastive specialists (e.g., CLIP/BASIC/LiT-style models) (Sec. 5, Appendix B.2).
  - In-context learning becomes less attractive at high shot counts due prompt sensitivity, compute cost, and weaker scaling of absolute gains (Sec. 5).
- **Generalization notes (OOD, compositionality, robustness):**
  - Performance consistently improves with model size and number of shots (Fig. 2).
  - Even though M3W training uses up to 5 images per sample, models benefit from prompts with up to 32 interleaved image/video examples at inference (Sec. 3.1).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - One model handles many open-ended image/video tasks via prompting.
  - Strong few-shot transfer with no per-task finetune on many benchmarks.
  - Modular architecture effectively reuses frozen pretrained backbones.
- **Failure cases / limitations (explicitly reported):**
  - Inherits LM weaknesses: hallucinations, ungrounded guesses, and sequence-length generalization issues (Sec. 5).
  - Weaker classification than dedicated contrastive models (Sec. 5).
  - In-context learning is sensitive to demonstrations/prompts and scales poorly in compute/performance beyond low-shot regime (Sec. 5).
  - Societal risks: offensive output, stereotype propagation, and privacy leakage from LM behavior, plus image-related bias risks (Sec. 5).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Moderate-to-high risk in open-ended generation when LM priors dominate weak visual evidence.
  - Weak grounding:
    - Can occur on visually ambiguous or compositional prompts, consistent with "ungrounded guesses" note.
  - OCR/text-in-image:
    - Mixed; TextVQA is included but model does not rely on explicit OCR pipeline by default.
  - Counting:
    - Likely brittle for precise counting due autoregressive prompting and weak explicit counting supervision.
  - Spatial reasoning:
    - Better than simple dual-encoders for open-ended QA, but still vulnerable on causal/temporal/spatial edge cases.
  - Evaluation artifact / benchmark weakness:
    - Prompt format, support-shot construction, and benchmark split choices can materially shift few-shot scores.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 2.1-2.2 / Fig. 3-4 / Pg. 4-5  
   - **Why it matters (1 line):** Establishes the key connector design that makes frozen-backbone multimodal conditioning workable.

2. **Claim:**  
   - **Where:** Sec. 3.1 / Table 1 / Pg. 7-8  
   - **Why it matters (1 line):** Supports the paper’s central few-shot claim across a broad benchmark set using a single model.

3. **Claim (optional):**  
   - **Where:** Sec. 3.3 / Table 3 / Pg. 8  
   - **Why it matters (1 line):** Shows performance depends strongly on data mixture and architectural choices (especially M3W and gating).

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - Flamingo is a connector-style visual language model that preserves pretrained unimodal knowledge by freezing both a strong vision backbone (NFNet-F6) and a large language model (up to Chinchilla 70B), while training only bridging components. The bridge has two parts: a Perceiver Resampler that compresses variable-sized image/video features into a fixed token set, and interleaved GATED XATTN-DENSE blocks inserted into the frozen LM stack. Training uses a single autoregressive next-token objective on a weighted mixture of web-scale multimodal corpora: interleaved webpage data (M3W) plus paired image-text (ALIGN, LTIP) and video-text (VTP) data. This combination is not cosmetic: ablations show that removing M3W or specific paired sources substantially hurts transfer. Inference is prompt-based few-shot learning over interleaved support examples, with open-ended generation for caption/VQA/dialogue and likelihood scoring for close-ended tasks. Empirically, Flamingo-80B achieves strong 32-shot performance across a 16-task suite and can exceed several fine-tuned specialists, while full supervised finetuning further improves tasks such as VQAv2 and COCO captioning. The paper is also clear about limits: Flamingo inherits LM hallucination and bias issues, remains prompt-sensitive in in-context mode, and still trails specialized contrastive models on classification-heavy settings.

## Table fields for the final comparison table
- **Model name + year:** Flamingo (2022)
- **Family:** connector
- **Vision backbone:** Frozen NFNet-F6 (contrastively pretrained)
- **Language backbone:** Frozen autoregressive LM family (1.4B/7B/70B Chinchilla-based; flagship is 70B)
- **Fusion/connector:** Perceiver Resampler (64 visual tokens) + interleaved GATED XATTN-DENSE blocks
- **Objective(s):** Weighted autoregressive next-token NLL over interleaved multimodal sequences
- **Data type/scale (high-level):** M3W (~43M webpages), ALIGN (1.8B pairs), LTIP (312M pairs), VTP (27M video-text pairs)
- **Key evals:** 16-task multimodal suite (VQA, captioning, dialogue, video QA, retrieval-like tasks) plus ImageNet/Kinetics and retrieval analyses in appendix
- **Known failure modes (1–3 bullets):**
  - LM-prior-driven hallucination/ungrounded responses
  - Prompt and demonstration sensitivity in in-context learning
  - Classification gap versus specialized contrastive models

## Notes / TODO
- **Open questions to verify later:**
  - Paper text alternates between "6" and "7" tasks beating fine-tuned SOTA in 32-shot mode; reconcile exact counting from official benchmark definitions.
  - Check whether any official DeepMind checkpoint/evaluation package was released after publication beyond what is linked on arXiv/NeurIPS.
- **Follow-up papers this cites (add to reading list):**
  - Chinchilla (Hoffmann et al., 2022)
  - CLIP (Radford et al., 2021)
  - ALIGN (Jia et al., 2021)
  - Perceiver (Jaegle et al., 2021)
  - Gato (Reed et al., 2022)
