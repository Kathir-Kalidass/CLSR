# Real-Time Vision-Based Continuous Sign Language Recognition and Translation (ISL)

## Executive Summary
This project designs and implements a real-time, camera-only system for continuous sign language recognition (CSLR) and translation focused on Indian Sign Language (ISL). The pipeline combines multi-feature visual processing (RGB + pose), temporal sequence modeling (BiLSTM/Transformer with CTC), and language-level correction to produce fluent English text and speech. The system is modular, scalable, and optimized for low-latency; it supports experimentation across datasets and model variants while targeting ISL for final evaluation.

**📘 For detailed implementation workflow, architecture, and step-by-step methodology, see [DETAILED_WORKFLOW.md](DETAILED_WORKFLOW.md).**

---

## 1. Problem Statement
Despite strong progress in sign language AI, typical solutions underperform in real-world ISL scenarios:
- Focus on isolated sign recognition rather than continuous sentence-level signing.
- Pretrained models and public datasets are predominantly ASL, not ISL.
- ISL has limited large-scale annotated datasets, hindering end-to-end training.
- Raw sign-to-text outputs lack grammatical correctness and semantic fluency.
- Real-time constraints add latency, alignment, and signer-robustness challenges.

We aim to build a robust, real-time system that:
- Targets ISL, using multimodal visual features.
- Handles unsegmented sign streams.
- Produces linguistically valid English output (text and speech).

---

## 2. Objectives
### Primary Objectives
- Real-time continuous sign language recognition using deep learning.
- Convert sign language video to English text and speech.
- Support continuous signing without manual segmentation.

### Secondary Objectives
- Multi-feature learning (RGB + pose/landmarks) for robustness.
- Temporal modeling (BiLSTM / Transformer) for sequence learning.
- AI-based sentence correction for grammatical validity.
- Evaluation via standard CSLR and translation metrics.

---

## 3. Base Paper & Influence
### 🎯 Primary Base Paper
**"Deep Learning-Based Sign Language Recognition Using Efficient Multi-Feature Attention Mechanism"**

Location: `report_pages/conference_journels_std/Deep_Learning-Based_Sign_Language_Recognition_Using_Efficient_Multi-Feature_Attention_Mechanism.pdf`

This paper serves as the **core architectural foundation** for our project and directly influences:

Reused Concepts:
- **Dual-stream feature extraction** (RGB + pose) — captures complementary visual information
- **Attention-based feature fusion** — adaptively weights modality importance
- **CNN-based spatial feature learning** — robust to appearance variations
- Performance evaluation methodology and metrics

Extensions & Contributions:
- Extend from **isolated** to **continuous** sign recognition
- Temporal modeling (BiLSTM/Transformer) + CTC decoding for unsegmented streams
- Language modeling and sentence correction for fluent output
- ISL-oriented pipeline with gloss replacement and fine-tuning
- Continuous sign segmentation, caption buffering, and reordering
- Integrated text-to-speech (TTS) for live feedback

**For complete base paper integration details, feature extraction workflows, and attention mechanism implementation, see [DETAILED_WORKFLOW.md](DETAILED_WORKFLOW.md) Section 2-3.**

---

## 4. System Architecture Overview
A modular, multi-stage pipeline optimized for real-time performance:

1) Video Ingestion & Preprocessing
- Frame sampling, resizing, normalization.
- Optional tracking, background handling, and denoising.

2) Multi-Feature Extraction (Dual Stream)
- RGB Stream: CNN/TSN/I3D-like backbone for spatial-temporal features.
- Pose Stream: Keypoints/landmarks (hands, body, face) via pose estimation; encoded with GCN/MLP/RNN.

3) Fusion & Temporal Modeling
- Attention-based fusion of RGB + pose features.
- Sequence modeling via BiLSTM or Transformer.
- Alignment with CTC for unsegmented streams.

4) Decoding, Buffering, and Reordering
- Beam search CTC decoding for gloss sequence.
- Sliding window buffers for partial predictions; reorder when confidence increases.

5) Language-Level Correction
- Grammar and fluency correction with lightweight LM or seq2seq.
- Domain-aware rules for ISL→English mapping where feasible.

6) Output
- Text: Continuous caption stream.
- Speech: TTS for real-time audio feedback.

7) Real-Time Orchestration
- Latency budgeting per stage; asynchronous queues.
- Robustness: signer variability, occlusions, lighting.

---

## 5. Dataset Strategy
### 5.1 Why Start with ASL Datasets
ASL resources (e.g., MS-ASL, ASLLVD, RWTH-PHOENIX) offer scale, diversity, and benchmarks.
- Validate architecture and training pipeline.
- Pretrain CNN/Transformer backbones and pose encoders.
- Learn generic gesture representations before ISL fine-tuning.

**Detailed justification:** ASL datasets provide a proven methodology for validating our dual-stream architecture from the base paper. Low-level visual features (hand shapes, motion patterns, pose dynamics) are largely **language-agnostic** and transfer effectively across sign languages through fine-tuning. This approach follows established transfer learning principles in computer vision research.

### 5.2 Migration from ASL to ISL (Feasible with Constraints)
**✅ YES — Technically and Practically Feasible**

Transferable:
- CNN spatial extractors (hand shape, motion patterns).
- Pose/landmark encoders (body skeleton structure is universal).
- Temporal layers (BiLSTM/Transformer) handle sequential patterns universally.
- CTC alignment and decoding logic is vocabulary-independent.
- **Attention mechanisms from base paper** are language-agnostic.

Non-transferable:
- Vocabulary (gloss labels differ between ASL and ISL).
- Grammar and sentence structure (ISL has different syntax).
- Cultural/linguistic sign variations (regional dialects).

Migration Approach:
1. **Phase 1:** Pretrain entire network on ASL datasets (MS-ASL, WLASL)
2. **Phase 2:** Replace classification head with ISL vocabulary
3. **Phase 3:** Fine-tune on ISL data (iSign DB) with reduced learning rate
4. **Phase 4:** Train ISL-specific language models for gloss-to-text translation

**For complete migration workflow, timeline, and phase-by-phase implementation plan, see [DETAILED_WORKFLOW.md](DETAILED_WORKFLOW.md) Section "ASL→ISL Migration".**

### 5.3 ISL Datasets We Can Use
Primary (Recommended)
- iSign DB: ISL–English paired dataset; sentence/phrase-level videos; multiple signers. Best for continuous CSLR and translation.

Supporting / Validation
- INCLUDE-50 ISL: ~50 common ISL gestures, isolated signs, controlled conditions. Good for early testing and module-level validation.
- ISL-CSLTR: Designed for continuous ISL recognition/translation; smaller than iSign DB; good for CTC and segmentation experiments.
- ISL Alphabet (Kaggle): Alphabet-only, isolated hand gestures. Useful for preprocessing/CNN/pose pipeline sanity checks.
- Custom ISL Dataset: Team-recorded samples for adaptability and real-time performance demos.

Supporting (Non-ISL)
- MS-ASL: Pretraining and architecture validation; large-scale ASL.
- RWTH-PHOENIX-Weather: Continuous signing for CTC/beam search validation.

### 5.4 Compact Comparison Table
| Dataset                | Type        | Scope                | Best Use                                | Notes                               |
|------------------------|-------------|----------------------|------------------------------------------|-------------------------------------|
| iSign DB               | Continuous  | ISL sentences/phrases| Final evaluation; end-to-end translation | ISL-focused; diverse signers         |
| INCLUDE-50 ISL         | Isolated    | ~50 common gestures  | Early-stage testing; module validation    | Clean background; limited vocab      |
| ISL-CSLTR              | Continuous  | ISL sentences        | CTC alignment; segmentation experiments   | Smaller than iSign DB               |
| ISL Alphabet (Kaggle)  | Isolated    | Alphabets (A–Z)      | Debugging preprocessing/CNN/pose          | Not sufficient for full CSLR        |
| Custom ISL             | Flexible    | Team-recorded        | Real-time demo; user-specific fine-tuning | Controlled environment               |
| MS-ASL (ASL)           | Continuous/Isolated | Large-scale ASL | Pretraining; backbone validation         | Transfer learning only               |
| RWTH-PHOENIX-Weather   | Continuous  | Weather broadcast    | CTC/beam search strategy validation       | Widely used CSLR benchmark           |

---

## 6. Methodology Details
### 6.1 Preprocessing
- Uniform frame sampling; normalization.
- Pose estimation (skeleton/hand keypoints) per frame.
- Optional noise handling and signer-invariance tricks.

### 6.2 Feature Extraction
- RGB encoder (CNN/I3D/TSM/SlowFast variants).
- Pose/landmark encoder (GCN/MLP/RNN).
- Parallel extraction for low latency.

### 6.3 Fusion & Temporal Modeling
- Attention-based fusion of streams.
- BiLSTM or Transformer encoders for sequence learning.
- CTC loss for unsegmented training.

### 6.4 Decoding & Language Correction
- CTC beam search with confidence thresholds.
- Lightweight LM for grammar correction; optional domain rules.
- Caption buffering to refine partial outputs.

### 6.5 TTS Integration
- Stream corrected text into TTS (offline/online).
- Audio feedback pipeline with minimal delay.

---

## 7. Evaluation
- Recognition: WER, CER, Top-1/Top-5 gloss accuracy.
- Translation: BLEU, ROUGE-L; human fluency ratings optional.
- Real-time: End-to-end latency, jitter, throughput.
- Robustness: Cross-signer generalization, occlusion sensitivity, lighting variance.

---

## 8. Experiments Plan
Phase A: Architecture Validation
- Pretrain on ASL (MS-ASL/RWTH-PHOENIX) to validate feature extractors and temporal models.

Phase B: ISL Adaptation
- Fine-tune on iSign DB; replace gloss dictionary and LMs.
- Evaluate on continuous ISL tasks; iterate fusion/temporal choices.

Phase C: Real-Time Deployment
- Integrate buffering, decoding, correction, TTS.
- Measure latency and robustness; demo custom ISL samples.

---

## 9. Implementation Map (Repo-Oriented)
- Reference pipelines and utilities:
  - references/NLA-SLR/: Data, configs, training, prediction utilities.
  - TwoStreamNetwork/: Dual-stream modeling, training, feature extraction.
  - Online/: Real-time components and requirements for streaming.
- Assets & Diagrams:
  - report_pages/: Architecture diagrams, slides, and presentation materials.

Use these folders to align experiments and documentation artifacts; keep ISL-specific configs isolated for clarity.

---

## 10. Setup & Run (Guidance)
Note: Adapt commands to the chosen pipeline; verify environment names.

Option A: Conda (example)
```bash
# From repo root
conda env create -f TwoStreamNetwork/environment.yml
conda activate cslr
```

Option B: Pip (example)
```bash
# Install per-module requirements (adjust as needed)
pip install -r references/NLA-SLR/requirements.txt
pip install -r Online/requirements.txt
```

Training & Prediction (examples; adjust paths/configs)
```bash
# NLA-SLR training
python references/NLA-SLR/training.py --config references/NLA-SLR/configs/nla_slr_msasl_100.yaml

# Two-stream training
python TwoStreamNetwork/training.py --config TwoStreamNetwork/experiments/example.yaml

# Prediction
python references/NLA-SLR/prediction.py --video /path/to/isl_video.mp4 --output outputs/
```

---

## 11. Risks & Mitigations
- Limited ISL data: Use transfer learning, data augmentation, and careful fine-tuning.
- Domain shift (ASL→ISL): Replace gloss dictionary and language models; prioritize ISL-specific evaluation.
- Grammar quality: Implement robust language correction and human-in-the-loop validation when needed.
- Real-time latency: Optimize feature extraction, use async queues, and tune window sizes.

---

## 12. References & Resources
### Primary Base Paper (MUST CITE)
📄 **"Deep Learning-Based Sign Language Recognition Using Efficient Multi-Feature Attention Mechanism"**
- File: `report_pages/conference_journels_std/Deep_Learning-Based_Sign_Language_Recognition_Using_Efficient_Multi-Feature_Attention_Mechanism.pdf`
- Core contributions used: Dual-stream architecture, attention-based fusion, multi-feature learning

### Supporting Research Papers
- "Toward Real-Time Recognition of Continuous Indian Sign Language: A Multi-Modal Approach Using RGB and Pose"
  - File: `report_pages/conference_journels_std/Toward_Real-Time_Recognition_of_Continuous_Indian_Sign_Language_A_Multi-Modal_Approach_Using_RGB_and_Pose.pdf`
- "Real-time Vision-based Indian Sign Language Translation Using Deep Learning Techniques"
  - File: `report_pages/conference_journels_std/Real-time Vision-based Indian Sign Language Translation Using Deep Learning Techniques.pdf`
- "iSign: A Benchmark for Indian Sign Language Process"
  - File: `report_pages/conference_journels_std/iSign_A_Benchmark_for_Indian_Sign_Language_Process.pdf`

### Architecture Diagrams
Visual documentation available in `report_pages/architecture_diagram/`:
- Overall system architecture
- Module-wise detailed diagrams (preprocessing, feature extraction, temporal modeling, output generation)

### Datasets
- MS-ASL: https://www.microsoft.com/en-us/research/project/ms-asl/
- WLASL: https://dxli94.github.io/WLASL/
- RWTH-PHOENIX-Weather: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
- iSign DB: Cite paper in `conference_journels_std/`

### Technical Resources
- CTC alignment and decoding literature
- Transformer-based sequence modeling references
- Spatial-Temporal GCN implementations
- MediaPipe Holistic for pose estimation

**For complete reference list with detailed citations and code repositories, see [DETAILED_WORKFLOW.md](DETAILED_WORKFLOW.md) Section "References & Further Reading".**

---

## 13. Appendices
- Glossary: CSLR, CTC, BiLSTM, Transformer, LM, TTS.
- Abbreviations: ISL, ASL, WER, CER, BLEU, ROUGE.
- Diagrams: See report_pages/architecture_diagram/ for visuals.

---

## Maintainer Notes
- Keep this report updated as datasets/configs evolve.
- Add concrete dataset stats (clips, hours, signers) when verified.
- Replace example commands with tested scripts and configs once finalized.
