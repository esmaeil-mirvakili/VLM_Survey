# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** Visual Instruction Tuning
- **Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **Venue / Year:** NeurIPS 2023 (Oral); arXiv v2 updated December 11, 2023
- **Link (PDF / arXiv):** https://arxiv.org/abs/2304.08485 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html
- **Code / Models (if any):** https://github.com/haotian-liu/LLaVA ; dataset: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - LLaVA is an early open-source connector-style LMM that applies instruction tuning to vision-language chat by pairing CLIP visual features with a Vicuna LLM and training on GPT-4-generated multimodal instruction data. It demonstrates strong multimodal chat behavior, 85.1% relative score vs GPT-4 on their synthetic instruction-following benchmark, and 92.53% ScienceQA accuracy when combined with GPT-4-as-judge.

## Positioning and scope

- **Problem addressed / motivation:**
  - Instruction tuning had boosted text-only LLM generalization, but was underexplored for multimodal models with image inputs. The paper aims to create both the data pipeline and model recipe for general-purpose visual instruction following.
- **What’s new vs prior work (1–3 bullets):**
  - First large-scale GPT-4-assisted visual instruction-tuning pipeline from image-caption/box context to conversations, detailed descriptions, and complex reasoning (Sec. 3, Table 1).
  - Simple but effective connector architecture (CLIP + linear projector + Vicuna) trained end-to-end for multimodal instruction following (Sec. 4.1, Fig. 1).
  - Introduces LLaVA-Bench (COCO + In-the-Wild) for instruction-following capability evaluation with GPT-4 judging protocol (Sec. 5.1, Tables 4-6).
- **Key assumptions (data, compute, setting):**
  - Assumes language-only GPT-4 can reliably generate high-quality multimodal instruction data from symbolic visual context (captions/boxes).
  - Assumes a lightweight projection connector is sufficient to transfer CLIP visual semantics into Vicuna token space.
  - Assumes text-only GPT-4 judging is a usable proxy for multimodal response quality (noted as a limitation in Appendix A).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: CLIP ViT-L/14 (open-set visual encoder); experiments consider features before/after last transformer layer (Sec. 4.1, 5.2 ablation).
  - Text/LLM backbone: Vicuna (mainly 13B in best setting; 7B also ablated) (Sec. 4.1, Table 8).
  - Fusion / connector mechanism: Single trainable linear projection matrix from CLIP feature space to LLM embedding space; no Q-Former/cross-attn module in core design (Sec. 4.1, Eq. 1).
  - Input representation (patches, regions, tokens, OCR, etc.): Projected visual tokens prepended with multi-turn instruction/assistant tokens in a chat-style sequence format (Table 2).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Primarily generation; for ScienceQA uses answer prediction with reasoning format and optional GPT-4 ensembling.
  - Uses prompting/instruction format? Y/N: Y
  - Any tool use (OCR, detector, retrieval index)? Y/N: Y (external GPT-4 used for data generation, benchmark judging, and one ensemble setting; not an in-model OCR/retrieval pipeline).

## Training objective(s)
- **Objective(s) used (list):**
  - Autoregressive language modeling loss over assistant response tokens in multimodal chat sequence (Sec. 4.2, Eq. 3).
- **Loss details (high-level, no math needed):**
  - The model predicts assistant responses conditioned on image tokens + accumulated chat history; training masks non-target tokens and optimizes standard next-token likelihood.
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - Stage-1 visual feature alignment uses filtered CC3M (CC-595K) converted into single-turn brief-description instructions.
    - Stage-2 visual instruction tuning uses LLaVA-Instruct-158K (58K conversation, 23K detailed description, 77K complex reasoning) generated via GPT-4 from COCO-derived context.
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - Yes: 2-stage procedure: feature alignment (freeze vision+LLM, train projector only) then end-to-end instruction tuning for chat behavior (Sec. 4.2).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - 8xA100 training; stage-1 LR 2e-3, BS 128, 1 epoch; stage-2 LR 2e-5, BS 32, 3 epochs; Adam (no weight decay), cosine schedule, warmup 3%, FSDP + gradient checkpointing, BF16/TF32 (Appendix C).

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - CC-595K filtered from CC3M for projector alignment; filtering preserves noun-phrase coverage while reducing noise (Sec. 4.2 + Appendix E/Fig. 7).
- **Finetuning / instruction data (if any):**
  - LLaVA-Instruct-158K generated with GPT-4 using COCO image captions and object boxes as text-only context; approximately 80K unique images (Sec. 3, Sec. 5.1).
  - ScienceQA downstream finetuning uses reasoning+answer formulation, 12 epochs in the reported best variant (Sec. 5.2).
- **Evaluation benchmarks (list all used):**
  - LLaVA-Bench (COCO): 30 images x 3 question types (conversation, detailed description, complex reasoning).
  - LLaVA-Bench (In-the-Wild): 24 curated images with 60 hard questions.
  - ScienceQA test benchmark with category breakdowns and several baselines/ensembling variants.
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - Relative score vs GPT-4 judge on LLaVA-Bench subsets; mean±std in repeated runs.
  - ScienceQA accuracy overall and by subject/context/grade splits.
- **Any ablations / diagnostic tests that matter:**
  - Data-type ablation on LLaVA-Bench (conversation-only vs +detailed/+reasoning) showing best performance with all three data types (Table 4).
  - ScienceQA ablations: CLIP layer choice, reasoning-first vs answer-first format, training from scratch vs two-stage pretraining, 7B vs 13B model scale (Table 8).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - Achieves 85.1% relative score to GPT-4 on LLaVA-Bench (COCO) with full 158K instruction data (Sec. 5.1, Table 4).
  - On ScienceQA, LLaVA+GPT-4(judge) reaches 92.53%, surpassing prior reported methods in that table (Sec. 5.2, Table 7).
- **Where it underperforms (tasks/conditions):**
  - LLaVA alone (90.92 ScienceQA) trails its own GPT-4-ensembled variant and can fail on fine-grained semantics/knowledge-heavy in-the-wild questions (Sec. 5.2; Table 6 examples).
  - Paper notes failure mode where model interprets image as a "bag of patches," missing compositional semantics in some cases (Sec. 5.1 limitations discussion).
- **Generalization notes (OOD, compositionality, robustness):**
  - Performs strongly on in-the-wild benchmark compared with BLIP-2 and OpenFlamingo under the authors’ GPT-4-based judging protocol (Table 5).
  - Shows emergent multi-turn chat behavior similar in style to multimodal GPT-4 examples, despite much smaller instruction-tuning data (Sec. 5.1, Table 3/9 examples).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Data-efficient connector approach with simple architecture and fast training cycle.
  - Strong instruction-following gains from synthetic multimodal instruction data.
  - Open release of model/code/data helped establish a practical multimodal baseline lineage.
- **Failure cases / limitations (explicitly reported):**
  - Hallucination and grounding failures inherited from base LLM behavior (Appendix A).
  - Bias transfer from both CLIP (vision) and LLaMA/Vicuna (language) (Appendix A).
  - Evaluation complexity and potential fragility of text-only GPT-4 judging across broader settings (Appendix A).
  - Challenges on high-resolution detail extraction and world-knowledge-heavy queries in LLaVA-Bench (Table 6 discussion).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Moderate-high risk in long answers and creative reasoning prompts, especially when visual evidence is weak.
  - Weak grounding:
    - Can miss compositional details (explicitly observed in "bag of patches" failure note).
  - OCR/text-in-image:
    - Limited in original LLaVA due fixed low-resolution CLIP features and no explicit OCR module.
  - Counting:
    - Vulnerable on dense scenes due simple linear connector and no region-aware explicit reasoning module.
  - Spatial reasoning:
    - Improved by GPT-4-generated reasoning data but still brittle in complex real-world edge cases.
  - Evaluation artifact / benchmark weakness:
    - Heavy reliance on GPT-4-as-judge and synthetic prompt templates can bias relative-score interpretation.

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. 3 / Tab. 1 / Pg. 3-4  
   - **Why it matters (1 line):** Establishes the GPT-4-assisted data generation recipe that enabled visual instruction tuning at practical scale.

2. **Claim:**  
   - **Where:** Sec. 5.1 / Tab. 4-5 / Pg. 7  
   - **Why it matters (1 line):** Quantifies that adding detailed and reasoning instruction data materially improves multimodal instruction-following quality.

3. **Claim (optional):**  
   - **Where:** Sec. 5.2 / Tab. 7 / Pg. 8-9  
   - **Why it matters (1 line):** Shows LLaVA is strong on ScienceQA and further gains from GPT-4-judge ensembling, highlighting complementary strengths.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - LLaVA (Visual Instruction Tuning) is an early connector-based large multimodal model that pairs a frozen CLIP ViT-L/14 visual encoder with a Vicuna language model via a lightweight linear projection. The training strategy is explicitly two-stage: first, feature alignment on filtered CC3M (CC-595K) while tuning only the projector; second, multimodal instruction tuning on LLaVA-Instruct-158K, a GPT-4-generated dataset with conversation, detailed description, and complex reasoning samples. Optimization uses standard autoregressive language modeling over assistant responses in a multi-turn chat format. Empirically, the paper reports major improvements in instruction-following behavior: on LLaVA-Bench (COCO), adding detailed and reasoning data to conversational training raises relative score substantially, culminating at 85.1% of a GPT-4 reference under their judging setup. On ScienceQA, LLaVA alone reaches 90.92% and the proposed LLaVA+GPT-4 judge ensemble reaches 92.53%. The paper also identifies limits that remain relevant: hallucinations and bias inherited from base models, dependence on evaluation protocol (GPT-4 judging), and failure on hard real-world compositional/knowledge-heavy queries where fine-grained perception or external knowledge is needed. Overall, LLaVA’s main contribution is less architectural novelty and more a practical recipe that made open multimodal instruction tuning reproducible and extensible.

## Table fields for the final comparison table
- **Model name + year:** LLaVA (Visual Instruction Tuning, 2023)
- **Family:** connector
- **Vision backbone:** CLIP ViT-L/14
- **Language backbone:** Vicuna (mainly 13B; 7B ablated)
- **Fusion/connector:** Single linear projection from visual features to LLM token embedding space
- **Objective(s):** Autoregressive language modeling on multimodal instruction-following sequences
- **Data type/scale (high-level):** CC-595K alignment + LLaVA-Instruct-158K GPT-4-generated multimodal instruction data
- **Key evals:** LLaVA-Bench (COCO/In-the-Wild), ScienceQA
- **Known failure modes (1–3 bullets):**
  - Hallucination / ungrounded outputs inherited from LLM base
  - Fine-grained compositional perception failures ("bag of patches" behavior)
  - Sensitivity to evaluation protocol and judgment heuristics

## Notes / TODO
- **Open questions to verify later:**
  - How stable are GPT-4-based comparative scores under updated GPT models and prompts?
  - Which later improvements (e.g., LLaVA-1.5/NeXT) contribute most: better data mixture, connector changes, or higher resolution vision input?
- **Follow-up papers this cites (add to reading list):**
  - Flamingo (Alayrac et al., 2022)
  - BLIP-2 (Li et al., 2023)
  - LLaMA-Adapter (Gao et al., 2023)
  - MM-CoT (Zhang et al., 2023)
  - Improved Baselines with Visual Instruction Tuning (Liu et al., 2023)
