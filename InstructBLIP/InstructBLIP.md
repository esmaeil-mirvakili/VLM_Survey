# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- **Authors:** Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi
- **Venue / Year:** NeurIPS 2023 (paper also released on arXiv in 2023)
- **Link (PDF / arXiv):** https://arxiv.org/abs/2305.06500 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html
- **Code / Models (if any):** Project code in LAVIS: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip ; example official checkpoints: https://huggingface.co/Salesforce/instructblip-flan-t5-xl and https://huggingface.co/Salesforce/instructblip-flan-t5-xxl

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - InstructBLIP extends BLIP-2 with instruction tuning at scale and an instruction-aware Q-Former that conditions visual feature extraction on the user instruction. Trained on 13 held-in datasets (from a 26-dataset pool), it reports SOTA zero-shot results on all 13 held-out datasets and strong finetuning transfer.

## Positioning and scope

- **Problem addressed / motivation:**
  - BLIP-2 and related connector models can generate language from images, but broad task generalization is limited when training mostly on caption-style supervision. The paper targets general-purpose VL instruction following across diverse tasks and domains.
- **What’s new vs prior work (1–3 bullets):**
  - Systematic VL instruction-tuning setup: 26 datasets, 11 task categories, split into 13 held-in and 13 held-out datasets (Sec. 2.1).
  - Instruction-aware visual extraction: instruction tokens are also fed to Q-Former (not only to LLM), so extracted visual features are task-aware (Sec. 2.3, Fig. 3).
  - Balanced multi-dataset sampling (sqrt-size with manual adjustments) to stabilize learning across heterogeneous datasets (Sec. 2.4).
- **Key assumptions (data, compute, setting):**
  - Assumes frozen pretrained vision encoder + LLM are strong enough; only Q-Former needs tuning for transfer (Sec. 2.6).
  - Assumes diverse instruction templates and dataset/task diversity improve unseen-task generalization (Sec. 2.1, 3.4).
  - Uses task-specific inference rules for some close-ended datasets (ranking candidates rather than free decoding), which affects direct comparability (Sec. 2.5).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: Frozen ViT-g/14 (inherited from BLIP-2 checkpoints) (Sec. 2.6).
  - Text/LLM backbone: Frozen FlanT5-XL / FlanT5-XXL or Vicuna-7B / Vicuna-13B (Sec. 2.6).
  - Fusion / connector mechanism: BLIP-2-style Q-Former bridge, modified to be instruction-aware (instruction tokens + learned queries interact via Q-Former self-attention), then projected as soft visual prompts for LLM (Sec. 2.3, Fig. 3).
  - Input representation (patches, regions, tokens, OCR, etc.): Image embeddings + instruction text; OCR tokens are appended for text-in-image tasks when available (Sec. 2.2, 2.5).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Hybrid. Free generation for open-ended tasks; constrained vocabulary ranking/log-likelihood for classification and multi-choice tasks (Sec. 2.5).
  - Uses prompting/instruction format? Y/N: Y
  - Any tool use (OCR, detector, retrieval index)? Y/N: Y (uses provided OCR tokens for relevant datasets; no detector/retrieval system).

## Training objective(s)
- **Objective(s) used (list):**
  - Instruction tuning uses standard language modeling loss to generate responses from image + instruction (Sec. 2.2).
  - Base model initialization comes from BLIP-2 two-stage pretraining before instruction tuning (Sec. 2.3).
- **Loss details (high-level, no math needed):**
  - The tuned model maximizes likelihood of target text responses under mixed-task instruction examples; at inference, some tasks switch from pure generation to candidate scoring for robust close-ended prediction (Sec. 2.5).
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - Mix all held-in training sets; sample instruction templates uniformly per dataset; sample datasets with sqrt-size probabilities + manual reweighting (e.g., lower A-OKVQA, higher OKVQA) (Sec. 2.4).
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - BLIP-2 pretraining (already done) -> InstructBLIP instruction tuning (Q-Former only) -> optional task-specific finetuning for downstream benchmarks (Sec. 2.3, 3.5).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - Max 60k steps, eval every 3k, batch sizes 192/128/64 (3B/7B/11-13B), AdamW (beta1 0.9, beta2 0.999, wd 0.05), LR warmup 1k steps from 1e-8 to 1e-5 then cosine decay; 16x A100-40GB, about 1.5 days/model (Sec. 2.6).

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - Uses BLIP-2 pretrained checkpoints as initialization. The instruction-tuning corpus itself spans 26 public datasets across 11 task categories; includes large Web CapFilt (14M image-text pairs) and LLaVA-Instruct-150K among held-in data (Sec. 2.1; Appendix Table 4).
- **Finetuning / instruction data (if any):**
  - Instruction tuning on 13 held-in datasets; zero-shot tested on 13 held-out datasets, including four fully held-out task categories (Sec. 2.1).
  - Additional task-specific finetuning experiments reported on ScienceQA, OCR-VQA, OKVQA, A-OKVQA (Sec. 3.5, Table 3).
- **Evaluation benchmarks (list all used):**
  - Held-out zero-shot set in Table 1: NoCaps, Flickr30K, GQA, VSR, IconQA, TextVQA, Visual Dialog, HatefulMemes, VizWiz, ScienceQA (image-context), MSVD-QA, MSRVTT-QA, iVQA.
  - Held-in aggregate eval includes COCO Caption, OKVQA, A-OKVQA, TextCaps (for averaging in ablations).
  - Task-specific finetuning evals: ScienceQA, OCR-VQA, OKVQA, A-OKVQA.
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - CIDEr (NoCaps/Flickr30K), iVQA accuracy, AUC (HatefulMemes), MRR (Visual Dialog), and top-1 accuracy for most other datasets (Table 1 note).
- **Any ablations / diagnostic tests that matter:**
  - Remove instruction-aware visual features: consistent held-in/held-out drops, especially on spatial/temporal reasoning tasks (Table 2, Sec. 3.2).
  - Remove balanced dataset sampling: less synchronized optimization and lower overall performance (Table 2, Sec. 3.2).
  - Instruction tuning vs multitask learning: instruction tuning materially improves held-out generalization, while multitask remains near BLIP-2 zero-shot baseline on held-out sets (Fig. 4, Sec. 3.4).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - Achieves SOTA zero-shot results on all 13 held-out datasets in their benchmark setup (Sec. 3.1, Table 1).
  - Smallest InstructBLIP FlanT5XL (~4B total) outperforms Flamingo-80B on all six shared evaluation datasets, with 24.8% average relative improvement (Sec. 3.1).
- **Where it underperforms (tasks/conditions):**
  - In downstream finetuning, it is still behind PaLM-E (562B) on OKVQA, despite beating BLIP-2 and setting SOTA on several other tasks (Sec. 3.5, Table 3).
  - Performance depends on LLM family: FlanT5 variants are better on multi-choice settings, while Vicuna variants are often stronger on open-ended generation (Sec. 3.5).
- **Generalization notes (OOD, compositionality, robustness):**
  - The key reported gain is unseen-task/data transfer: instruction tuning improves held-out zero-shot behavior more than multitask training (Sec. 3.4, Fig. 4).
  - It reports strong gains on unseen video-QA task categories despite no temporal video training in instruction tuning stage (Sec. 3.1).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Strong zero-shot generalization across heterogeneous held-out benchmarks.
  - Efficient adaptation strategy: keeps vision encoder and LLM frozen; tunes only Q-Former.
  - Good initialization for downstream finetuning, consistently improving over BLIP-2 in Table 3.
- **Failure cases / limitations (explicitly reported):**
  - Inherits frozen LLM failure modes, including hallucinated ungrounded text and biased outputs (Appendix A: Broader Impact).
  - Authors explicitly caution against deployment without safety/fairness assessment for the target application (Appendix A).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Moderate risk remains due frozen LLM priors; instruction-aware grounding reduces but does not remove it.
  - Weak grounding:
    - Lower than BLIP-2 on many tasks, but still likely in fine-grained or ambiguous scenes.
  - OCR/text-in-image:
    - Better than generic caption-only tuning because OCR tokens are integrated for relevant datasets.
  - Counting:
    - Likely still brittle on precise counting unless explicitly represented in held-in instruction data.
  - Spatial reasoning:
    - Improved by instruction-aware Q-Former (ablation supports this), but still error-prone on complex compositions.
  - Evaluation artifact / benchmark weakness:
    - Heavy dependence on instruction template design and candidate-ranking setup can inflate/deflate task-specific scores.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 3.1 / Tab. 1 / Pg. 6-7  
   - **Why it matters (1 line):** Core empirical claim: instruction tuning + instruction-aware Q-Former yields broad held-out zero-shot gains over BLIP-2/Flamingo baselines.

2. **Claim:**  
   - **Where:** Sec. 3.4 / Fig. 4 / Pg. 8  
   - **Why it matters (1 line):** Isolates that instruction tuning, not plain multitasking, drives unseen-task generalization improvements.

3. **Claim (optional):**  
   - **Where:** Sec. 3.2 / Tab. 2 / Pg. 7  
   - **Why it matters (1 line):** Shows instruction-aware visual feature extraction and balanced data sampling are causal contributors, not incidental changes.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - InstructBLIP is a connector-style extension of BLIP-2 that focuses on improving instruction-following generalization in vision-language settings. The key architectural change is an instruction-aware Q-Former: instead of extracting one fixed visual representation per image, Q-Former also receives instruction tokens so visual features can adapt to the task request before being passed as soft prompts to a frozen LLM. Training is framed as vision-language instruction tuning with standard language-modeling loss over a diverse multi-task mixture (26 datasets spanning 11 categories), with a careful 13 held-in/13 held-out protocol to test real zero-shot transfer. The paper also introduces balanced dataset sampling (sqrt-size plus manual reweighting) to prevent overfitting small datasets and under-training large ones. Empirically, this combination improves held-out zero-shot performance consistently over BLIP-2 across FlanT5 and Vicuna backbones, and the smallest FlanT5XL variant reportedly beats Flamingo-80B on all shared tasks in their setup. Ablations show performance drops when instruction-aware feature extraction or balanced sampling is removed, especially on spatial/temporal reasoning benchmarks. Even so, limitations remain: because the LLM is frozen, the model still inherits LLM-level hallucination and bias risks, and the authors recommend task-specific safety/fairness assessment before deployment.

## Table fields for the final comparison table
- **Model name + year:** InstructBLIP (2023)
- **Family:** connector
- **Vision backbone:** Frozen ViT-g/14 (BLIP-2 image encoder)
- **Language backbone:** Frozen FlanT5-XL/XXL or Vicuna-7B/13B
- **Fusion/connector:** Instruction-aware Q-Former + linear projection as soft visual prompts
- **Objective(s):** Instruction tuning with language modeling loss (Q-Former-only finetuning)
- **Data type/scale (high-level):** 26 public VL datasets (11 task categories); 13 held-in instruction-tuning datasets and 13 held-out zero-shot datasets
- **Key evals:** Held-out 13-dataset zero-shot suite + downstream finetuning on ScienceQA/OCR-VQA/OKVQA/A-OKVQA
- **Known failure modes (1–3 bullets):**
  - Residual hallucination and bias inherited from frozen LLMs
  - Prompt/template and candidate-set sensitivity for close-ended tasks
  - Relative weakness on some benchmarks versus much larger specialized models (e.g., OKVQA vs PaLM-E in finetuned setting)

## Notes / TODO
- **Open questions to verify later:**
  - Replication sensitivity to instruction templates and manual dataset reweighting choices.
  - How much of gains come from instruction-aware Q-Former vs broader/more diverse data alone at fixed compute.
- **Follow-up papers this cites (add to reading list):**
  - BLIP-2 (Li et al., 2023)
  - Flamingo (Alayrac et al., 2022)
  - FLAN / FlanT5 (Chung et al., 2022)
  - LLaMA (Touvron et al., 2023)
  - LLaVA (Liu et al., 2023)
