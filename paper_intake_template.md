# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:**
- **Authors:**
- **Venue / Year:**
- **Link (PDF / arXiv):**
- **Code / Models (if any):**

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - 

## Positioning and scope

- **Problem addressed / motivation:**
  -
- **What’s new vs prior work (1–3 bullets):**
  - 
- **Key assumptions (data, compute, setting):**
  - 

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder:
  - Text/LLM backbone:
  - Fusion / connector mechanism:
  - Input representation (patches, regions, tokens, OCR, etc.):
- **Training/inference style:**
  - Retrieval vs generation vs hybrid:
  - Uses prompting/instruction format? Y/N
  - Any tool use (OCR, detector, retrieval index)? Y/N

## Training objective(s)
- **Objective(s) used (list):**
  - 
- **Loss details (high-level, no math needed):**
  - 
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
  - Multi-stage training? (pretrain → finetune → instruction tuning):
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - 
- **Finetuning / instruction data (if any):**
  - 
- **Evaluation benchmarks (list all used):**
  - 
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - 
- **Any ablations / diagnostic tests that matter:**
  - 

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - 
- **Where it underperforms (tasks/conditions):**
  - 
- **Generalization notes (OOD, compositionality, robustness):**
  - 

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - 
- **Failure cases / limitations (explicitly reported):**
  - 
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
  - Weak grounding:
  - OCR/text-in-image:
  - Counting:
  - Spatial reasoning:
  - Evaluation artifact / benchmark weakness:

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Sec. __ / Fig. __ / Tab. __ / Appx __ / Pg. __  
   - **Why it matters (1 line):**  

2. **Claim:**  
   - **Where:** Sec. __ / Fig. __ / Tab. __ / Appx __ / Pg. __  
   - **Why it matters (1 line):**  

3. **Claim (optional):**  
   - **Where:** Sec. __ / Fig. __ / Tab. __ / Appx __ / Pg. __  
   - **Why it matters (1 line):**  

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - 

## Table fields for the final comparison table
- **Model name + year:**
- **Family:** dual-encoder / cross-attn fusion / connector
- **Vision backbone:**
- **Language backbone:**
- **Fusion/connector:**
- **Objective(s):**
- **Data type/scale (high-level):**
- **Key evals:**
- **Known failure modes (1–3 bullets):**
  - 

## Notes / TODO
- **Open questions to verify later:**
  - 
- **Follow-up papers this cites (add to reading list):**
  - 
