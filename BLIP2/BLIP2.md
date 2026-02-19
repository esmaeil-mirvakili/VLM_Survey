# Paper Intake Template (Vision-Language Models)

## Bibliographic info
- **Title:** BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- **Authors:** Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi
- **Venue / Year:** ICML 2023 (PMLR 202:19730-19742)
- **Link (PDF / arXiv):** https://proceedings.mlr.press/v202/li23q/li23q.pdf ; https://arxiv.org/abs/2301.12597
- **Code / Models (if any):** https://github.com/salesforce/LAVIS/tree/main/projects/blip2 ; model releases: https://huggingface.co/Salesforce/blip2-opt-2.7b , https://huggingface.co/Salesforce/blip2-flan-t5-xl

## TL;DR contribution (1–2 sentences)
- **Contribution:**
  - BLIP-2 introduces a lightweight connector (Q-Former) that bridges a frozen vision encoder and a frozen LLM via two-stage pretraining, avoiding expensive end-to-end multimodal training. It reports strong zero-shot VQA/caption/retrieval results, including +8.7% over Flamingo80B on VQAv2 with 54x fewer trainable parameters (Abstract, Table 2).

## Positioning and scope

- **Problem addressed / motivation:**
  - End-to-end VLP has become very expensive; the paper targets compute-efficient transfer by reusing frozen unimodal backbones and training only a small bridging module (Sec. 1).
- **What’s new vs prior work (1–3 bullets):**
  - A Querying Transformer (Q-Former) as an information bottleneck between frozen image and language models (Sec. 3.1, Fig. 2).
  - Two-stage pretraining: representation learning with frozen vision model, then generative alignment with frozen LLM (Sec. 3.2-3.3, Fig. 1/3).
  - Strong zero-shot and finetuned results across VQA, captioning, and retrieval with far fewer trainable parameters than large end-to-end baselines (Tables 1-5).
- **Key assumptions (data, compute, setting):**
  - Assumes high-quality pretrained unimodal models are available and remain frozen.
  - Relies on web-scale image-text pretraining (129M images total, including 115M LAION-derived images with CapFilt) (Sec. 3.4).
  - Assumes pairwise image-text training data (single pair/sample), which later appears to limit in-context learning (Sec. 5).

## Model type and architecture
- **Architecture type (pick one):** `dual-encoder` / `cross-attn fusion` / `connector`
- **High-level design:**
  - Vision encoder: Frozen ViT-L/14 (CLIP) or ViT-g/14 (EVA-CLIP); second-last layer visual features are used (Sec. 3.4).
  - Text/LLM backbone: Frozen OPT (decoder LLMs) and FlanT5 (encoder-decoder LLMs) variants (Sec. 3.3-3.4).
  - Fusion / connector mechanism: Q-Former initialized from BERTbase with inserted cross-attention every other block; 32 learnable queries (dim 768), 188M params total for Q-Former (Sec. 3.1, Fig. 2).
  - Input representation (patches, regions, tokens, OCR, etc.): Image tokens from frozen ViT; query tokens extract compressed visual representation Z; text tokens interact via task-specific masks (uni-modal / bi-directional / multimodal causal) (Sec. 3.1-3.2).
- **Training/inference style:**
  - Retrieval vs generation vs hybrid: Hybrid (contrastive/matching/generation objectives in pretraining; generation and retrieval at downstream time).
  - Uses prompting/instruction format? Y/N: Y (e.g., VQA prompts and instructed image-to-text generation in Sec. 4.1).
  - Any tool use (OCR, detector, retrieval index)? Y/N: N

## Training objective(s)
- **Objective(s) used (list):**
  - Stage 1: ITC (image-text contrastive), ITM (image-text matching), ITG (image-grounded text generation) (Sec. 3.2).
  - Stage 2: Language modeling objective for decoder-only LLMs (OPT) and prefix language modeling for encoder-decoder LLMs (FlanT5) (Sec. 3.3).
- **Loss details (high-level, no math needed):**
  - ITC aligns query-based image representation with text [CLS] embedding using in-batch negatives and max-sim over queries. ITM performs binary matched/unmatched classification with hard negatives. ITG enforces query-to-text information flow through causal masking. Stage 2 trains Q-Former outputs (after linear projection) to become consumable by frozen LLMs for text generation (Sec. 3.2-3.3).
- **Training recipe highlights (only what matters):**
  - Data mixture strategy:
    - BLIP dataset (129M): COCO, Visual Genome, CC3M, CC12M, SBU, plus 115M LAION400M images. CapFilt generates 10 synthetic captions and keeps top-2 per image by CLIP score (Sec. 3.4).
  - Multi-stage training? (pretrain → finetune → instruction tuning):
    - Two-stage pretrain (representation then generative), followed by task-specific finetuning for caption/VQA/retrieval in experiments (Sec. 3.2-3.4, Sec. 4.2-4.4).
  - Notable hyperparams or tricks (temperature, hard negatives, caption filtering, etc.):
    - 250k (stage 1) + 80k (stage 2) steps; AdamW, cosine decay, warmup 2k; in-batch negatives; hard negative mining for ITM; mixed precision; largest model reported trainable on a single 16xA100(40GB) machine in <6 days + <3 days for two stages (Sec. 3.4).

## Data, datasets, and evaluation
- **Pretraining data (what + scale, high-level):**
  - 129M image-text examples total: curated sets + large web subset from LAION400M with CapFilt augmentation/filtering (Sec. 3.4).
- **Finetuning / instruction data (if any):**
  - Captioning: finetune on COCO, evaluate COCO + zero-shot transfer to NoCaps (Sec. 4.2).
  - VQA: finetune with VQAv2 train+val plus Visual Genome train samples (Sec. 4.3).
  - Retrieval: finetune on COCO, evaluate on COCO and zero-shot/transfer on Flickr30K (Sec. 4.4).
- **Evaluation benchmarks (list all used):**
  - Zero-shot VQA: VQAv2, OK-VQA, GQA (Table 2).
  - Image captioning: NoCaps (in/near/out/overall), COCO Karpathy test (Table 3).
  - VQA finetuning: VQAv2 val/test-dev (Table 4).
  - Image-text retrieval: Flickr30K zero-shot and COCO fine-tuned retrieval (Table 5).
- **Metrics reported (e.g., accuracy, CIDEr, recall@K, human eval):**
  - VQA accuracy; CIDEr/SPICE/BLEU@4 for captioning; Recall@K (R@1/R@5/R@10) for retrieval.
- **Any ablations / diagnostic tests that matter:**
  - Representation-learning ablation: removing stage-1 representation learning significantly hurts zero-shot VQA and can cause catastrophic forgetting (Fig. 5).
  - Scaling trends: stronger vision encoder and stronger LLM both improve VQA (Table 2 discussion).
  - Retrieval ablation: adding ITG to ITC+ITM improves retrieval scores (Table 6).

## Results summary (what actually moved)
- **Best headline result (1–2 bullets):**
  - Zero-shot VQAv2 test-dev: 65.0 (ViT-g + FlanT5XXL), +8.7 over Flamingo80B (56.3), while using 54x fewer trainable parameters (Abstract, Table 2).
  - Strong zero-shot caption/retrieval in the summary table: NoCaps CIDEr 121.6 and Flickr TR@1/IR@1 of 97.6/89.7 (Table 1).
- **Where it underperforms (tasks/conditions):**
  - On OK-VQA, BLIP-2 is below Flamingo80B; authors attribute this to open-world knowledge demands and stronger 70B LLM knowledge in Flamingo (Sec. 4.1 discussion around Table 2).
  - In-context VQA prompting did not improve performance in their setup (Sec. 5).
- **Generalization notes (OOD, compositionality, robustness):**
  - Shows strong transfer to out-of-domain NoCaps splits after COCO finetuning (Table 3, Sec. 4.2).
  - Demonstrates emergent prompted image-to-text behaviors (visual conversation/reasoning examples), but these are qualitative and can fail (Fig. 4, Fig. 7, Sec. 5).

## Strengths and failure cases (from the paper)
- **Strengths (supported by evidence in paper):**
  - Efficient: trains only connector module during pretraining while leveraging frozen strong unimodal backbones (Sec. 1, Sec. 3).
  - Versatile: same framework supports retrieval, captioning, and VQA with competitive/SOTA results (Tables 1-5).
  - Scales with better unimodal components (Table 2 analysis).
- **Failure cases / limitations (explicitly reported):**
  - Lacks observed in-context learning gains for VQA, likely due to single image-text pair training format (Sec. 5).
  - Generation can be unsatisfactory from incorrect LLM knowledge, wrong reasoning path, or stale world knowledge (Sec. 5; Fig. 7).
  - Inherits frozen LLM risks: offensive language, social bias propagation, privacy leakage (Sec. 5).
- **My read: likely failure tendencies (grounded guess, label clearly):**
  - Hallucination:
    - Moderate risk through LLM decoding, especially in knowledge-heavy prompts (consistent with Sec. 5).
  - Weak grounding:
    - Possible when Q-Former bottleneck misses fine-grained visual details or prompt steers toward priors.
  - OCR/text-in-image:
    - Not an explicit focus; likely variable unless pretrained encoder/LLM implicitly captures text cues.
  - Counting:
    - Likely brittle on precise counting given generative VQA tendencies and weak explicit counting supervision.
  - Spatial reasoning:
    - Better than naive prompting, but still prone to errors in complex multi-object relations.
  - Evaluation artifact / benchmark weakness:
    - Prompt template and decoding settings materially influence zero-shot VQA scores (Sec. 4.1 details).

## 2–3 “quotable” claims to paraphrase (with section pointers)
> Don’t copy text verbatim. Paraphrase and cite the exact location.

1. **Claim:**  
   - **Where:** Abstract + Sec. 4.1 / Tab. 2 / Pg. 1, 5  
   - **Why it matters (1 line):** Quantifies BLIP-2’s main efficiency-performance claim against Flamingo80B.

2. **Claim:**  
   - **Where:** Sec. 3.1-3.3 / Fig. 2-3 / Pg. 2-3  
   - **Why it matters (1 line):** Establishes the core connector design and why two-stage training is needed to bridge modalities.

3. **Claim (optional):**  
   - **Where:** Sec. 5 / Fig. 7 / Pg. 7  
   - **Why it matters (1 line):** Clarifies practical limits and safety risks that affect downstream deployment decisions.

## Drop-in writeup (for your report)
- **Target section:** Architecture / Training Objectives / Failure Modes-Eval
- **150–250 word paragraph draft:**
  - BLIP-2 is a connector-style VLM that avoids end-to-end multimodal pretraining by freezing both the vision encoder and the LLM, and only training a bridging Q-Former. The connector uses learnable query tokens and cross-attention (inserted every other transformer block) to compress image features into a small set of query embeddings, which are then projected into the language model input space. Training is explicitly two-stage. Stage 1 learns image-language alignment with three objectives (ITC, ITM, ITG), each with different attention masks so the same Q-Former can support contrastive alignment, pair discrimination, and grounded generation behavior. Stage 2 then aligns those query embeddings to a frozen decoder-only or encoder-decoder LLM using language modeling or prefix language modeling. This modular design delivers strong results with far fewer trainable parameters than contemporary large models: notably 65.0 zero-shot VQAv2 test-dev and an 8.7-point lead over Flamingo80B while using 54x fewer trainable parameters. The paper also reports strong captioning and retrieval transfer, including NoCaps gains and high Flickr retrieval recall. Reported limitations matter: no clear in-context learning gains, occasional unsatisfactory generations from wrong reasoning/knowledge, and inherited LLM safety/bias/privacy risks. So BLIP-2 is best viewed as a highly efficient transfer scaffold rather than a full solution to grounded reasoning reliability.

## Table fields for the final comparison table
- **Model name + year:** BLIP-2 (2023)
- **Family:** connector
- **Vision backbone:** Frozen ViT-L/14 (CLIP) or ViT-g/14 (EVA-CLIP)
- **Language backbone:** Frozen OPT (2.7B/6.7B) and FlanT5 (XL/XXL)
- **Fusion/connector:** Q-Former (BERTbase-init, 32 learned queries, cross-attn every other block) + linear projection to LLM
- **Objective(s):** ITC + ITM + ITG (stage 1), LM/prefix-LM with frozen LLM (stage 2)
- **Data type/scale (high-level):** 129M image-text pairs (COCO/VG/CC3M/CC12M/SBU + 115M LAION-derived) with CapFilt caption bootstrapping
- **Key evals:** VQAv2/OK-VQA/GQA (zero-shot), NoCaps + COCO captioning, VQAv2 finetune, Flickr30K/COCO retrieval
- **Known failure modes (1–3 bullets):**
  - Weak in-context learning gains in VQA setup
  - Hallucinated/incorrect generations from LLM knowledge/reasoning failures
  - Inherited LLM safety risks (offensive output, social bias, privacy leakage)

## Notes / TODO
- **Open questions to verify later:**
  - How much of gains come from CapFilt data bootstrapping versus architecture/training objectives?
  - Whether later BLIP-2 derivatives (e.g., instruction-tuned variants) fix the in-context learning gap noted in Sec. 5.
- **Follow-up papers this cites (add to reading list):**
  - Flamingo (Alayrac et al., 2022)
  - Frozen (Tsimpoukelli et al., 2021)
  - BLIP (Li et al., 2022)
  - CLIP (Radford et al., 2021)
  - LiT (Zhai et al., 2022)
