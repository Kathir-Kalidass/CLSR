# System Architecture Overview

## Document Purpose
This document provides a high-level overview of the complete system architecture for the Real-Time Vision-Based Continuous Indian Sign Language Recognition and Translation System. For detailed modular flows, see [03_complete_modular_flow.md](03_complete_modular_flow.md).

---

## Training & Deployment Architecture

### Training Environment: Google Colab
- **GPU Acceleration:** Tesla T4/V100/A100 (free tier available)
- **Dataset Storage:** Google Drive mounted for iSign DB access
- **Training Pipeline:** Full model training with GPU compute
- **Sliding-Window Inference:** Efficient continuous sequence processing (64-frame windows)
- **Checkpointing:** Automatic save to Google Drive
- **Monitoring:** TensorBoard integration for loss/metrics tracking

### Deployment Environment: Local System
- **Webcam Access:** Real-time video capture (cv2.VideoCapture)
- **CPU Inference:** Lightweight inference without GPU requirement
- **Browser Limitation:** Colab cannot access local webcam due to sandbox restrictions
- **Model Loading:** Download trained checkpoint from Colab/Drive
- **Real-Time Processing:** <500ms latency with sliding-window approach

### Why This Split?
1. **Training requires GPU** (hours of compute) → Colab provides free GPU
2. **Deployment requires webcam** (local hardware) → Must run on local machine
3. **Sliding-window enables real-time continuous recognition** without full buffering
4. **Cost-effective:** Free Colab GPU + consumer-grade local CPU

---

## 1. System Architecture Philosophy

The proposed architecture follows **five core design principles**:

### 1.1 Modularity
- Each component solves a specific sub-problem
- Independent development and testing
- Easy replacement of individual modules

### 1.2 Scalability
- Supports multiple datasets (ASL, ISL)
- Configurable vocabulary sizes (100 to 2000+ glosses)
- Adaptable to different sign languages

### 1.3 Real-Time Performance
- Target latency: **<500ms** end-to-end
- Asynchronous processing pipelines
- Optimized inference (quantization, pruning)

### 1.4 Robustness
- Multi-feature learning (RGB + Pose)
- Attention-based adaptive fusion
- Handles variations in lighting, background, signer appearance

### 1.5 Research Alignment
- Based on established base paper ("Multi-Feature Attention Mechanism")
- Extends with state-of-the-art temporal modeling
- Follows transfer learning best practices

---

## 2. Global System Architecture

The architecture is organized into **four major functional layers**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: INPUT & PREPROCESSING                │
│  • Video acquisition (camera/file)                               │
│  • Frame extraction and normalization                            │
│  • Pose/landmark extraction                                      │
│  • Temporal standardization                                      │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            LAYER 2: FEATURE EXTRACTION & FUSION                  │
│  • RGB Stream: CNN-based spatial features                        │
│  • Pose Stream: Keypoint-based motion features                   │
│  • Attention-Based Fusion (Base Paper Core)                      │
│  • Fused multi-modal representation                              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          LAYER 3: CONTINUOUS SIGN RECOGNITION                    │
│  • Temporal Modeling (BiLSTM/Transformer)                        │
│  • CTC Alignment (unsegmented sequences)                         │
│  • Decoding Strategy (Greedy/Beam Search)                        │
│  • ISL Gloss Sequence Generation                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          LAYER 4: LANGUAGE PROCESSING & OUTPUT                   │
│  • Caption buffering and merging                                 │
│  • Seq2Seq translation (ISL gloss → English)                     │
│  • Grammar correction and refinement                             │
│  • Text display + Text-to-Speech synthesis                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Data Flow

### 3.1 Input to Output Pipeline

```
ISL Signer → Camera Capture
                ↓
        [RGB Video Stream]
                ↓
        ┌───────────────┐
        │ PREPROCESSING │
        └───────┬───────┘
                ↓
    ┌───────────────────────┐
    │  Normalized RGB Frames │
    │  + Pose Keypoints      │
    └───────┬───────────────┘
            ↓
    ┌─────────────────────────────┐
    │  DUAL-STREAM EXTRACTION     │
    │  • RGB: CNN (ResNet/I3D)    │
    │  • Pose: GCN/MLP/RNN        │
    └───────┬─────────────────────┘
            ↓
    ┌──────────────────────────────┐
    │  ATTENTION-BASED FUSION      │
    │  (Base Paper Contribution)   │
    └───────┬──────────────────────┘
            ↓
    ┌──────────────────────────────┐
    │  TEMPORAL MODELING           │
    │  • BiLSTM or Transformer     │
    │  • CTC Alignment             │
    └───────┬──────────────────────┘
            ↓
    ┌──────────────────────────────┐
    │  DECODING                    │
    │  • Greedy / Beam Search      │
    └───────┬──────────────────────┘
            ↓
    [ISL Gloss Sequence]
            ↓
    ┌──────────────────────────────┐
    │  LANGUAGE TRANSLATION        │
    │  • Seq2Seq (Gloss → Text)    │
    │  • Grammar Correction        │
    └───────┬──────────────────────┘
            ↓
    [English Sentence]
            ↓
    ┌──────────────────────────────┐
    │  TEXT-TO-SPEECH              │
    │  • TTS Engine                │
    └───────┬──────────────────────┘
            ↓
    Text Display + Audio Output
```

### 3.2 Data Representations at Each Stage

| Stage | Data Representation | Dimensions |
|-------|---------------------|------------|
| **Input** | RGB video frames | (T × H × W × 3) |
| **Preprocessing** | Normalized frames + keypoints | (T × 224 × 224 × 3) + (T × N_kp × D) |
| **RGB Features** | CNN embeddings | (T × D_rgb) |
| **Pose Features** | Encoded keypoints | (T × D_pose) |
| **Fused Features** | Attention-weighted | (T × D_fusion) |
| **Temporal Features** | Contextualized | (T × D_model) |
| **CTC Output** | Gloss probabilities | (T × |Vocab|) |
| **Decoded** | Gloss sequence | List of gloss tokens |
| **Translated** | English text | String |
| **Audio** | Speech waveform | Audio array |

---

## 4. Module Interaction Diagram

```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │ video stream
       ▼
┌─────────────────────┐
│  Video Capture &    │◄─── Config: FPS, Resolution
│  Frame Extraction   │
└──────┬─────┬────────┘
       │     │
       │     └─────────────┐
       │                   │
       ▼                   ▼
┌──────────────┐    ┌─────────────────┐
│ RGB Frame    │    │ Pose Estimation │
│ Normalization│    │ (MediaPipe/     │
│              │    │  OpenPose)      │
└──────┬───────┘    └─────────┬───────┘
       │                      │
       │ RGB tensors          │ Keypoint tensors
       │                      │
       ▼                      ▼
┌──────────────┐    ┌─────────────────┐
│  CNN Backbone│    │  Pose Encoder   │
│ (ResNet/I3D) │    │  (GCN/MLP/RNN)  │
└──────┬───────┘    └─────────┬───────┘
       │                      │
       │ RGB features         │ Pose features
       │                      │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │  Attention Fusion    │◄─── Base Paper Core
       │  • Temporal Attn     │
       │  • Cross-modal Attn  │
       └──────────┬───────────┘
                  │ Fused features
                  ▼
       ┌──────────────────────┐
       │  Temporal Encoder    │
       │  • BiLSTM / Trans.   │
       └──────────┬───────────┘
                  │ Contextualized features
                  ▼
       ┌──────────────────────┐
       │     CTC Layer        │
       └──────────┬───────────┘
                  │ Frame-level probs
                  ▼
       ┌──────────────────────┐
       │  CTC Decoder         │◄─── Greedy or Beam Search
       │  (Greedy/Beam)       │
       └──────────┬───────────┘
                  │ Gloss sequence
                  ▼
       ┌──────────────────────┐
       │  Caption Buffer      │
       │  & Token Accumulator │
       └──────────┬───────────┘
                  │ Buffered glosses
                  ▼
       ┌──────────────────────┐
       │  Seq2Seq Translator  │
       │  (Gloss → English)   │
       └──────────┬───────────┘
                  │ Raw English text
                  ▼
       ┌──────────────────────┐
       │  Grammar Correction  │
       │  & Refinement        │
       └──────────┬───────────┘
                  │ Corrected text
                  ├────────────────┐
                  │                │
                  ▼                ▼
       ┌──────────────┐   ┌──────────────┐
       │ Text Display │   │  TTS Engine  │
       └──────────────┘   └──────┬───────┘
                                 │ Audio
                                 ▼
                          ┌──────────────┐
                          │   Speaker    │
                          └──────────────┘
```

---

## 5. Core Architectural Components

### 5.1 Base Paper Components (Reused)

#### 5.1.1 Dual-Stream Feature Extraction
**From:** "Deep Learning-Based Sign Language Recognition Using Efficient Multi-Feature Attention Mechanism"

**RGB Stream:**
- Captures appearance, texture, color information
- CNN backbone (ResNet-18/50, I3D, C3D)
- Pretrained on ImageNet/Kinetics
- Output: Spatial-temporal features

**Pose Stream:**
- Captures geometric structure and motion
- Input: 2D/3D keypoints (body, hands, face)
- Encoder: ST-GCN, MLP, or RNN
- Output: Skeletal motion features

**Why Dual-Stream?**
- Complementary information
- RGB: Robust to skeletal tracking errors
- Pose: Robust to appearance variations
- Combined: Higher accuracy (+5-8% from base paper)

#### 5.1.2 Attention-Based Fusion
**From:** Base paper's core innovation

**Mechanism:**
1. **Temporal Self-Attention** (per modality)
   - Identify important time steps
   - Focus on discriminative frames
   - Reduce noise from transition frames

2. **Cross-Modal Attention**
   - Adaptive weighting of RGB vs Pose
   - Context-dependent fusion
   - Formula: `F_fused = α(t) × F_rgb(t) + β(t) × F_pose(t)`
   - α, β learned per frame

**Advantages:**
- Outperforms simple concatenation
- Adapts to input quality (e.g., poor lighting → rely more on pose)
- Interpretable (attention weights show modality importance)

### 5.2 Extended Components (Our Contributions)

#### 5.2.1 Temporal Sequence Modeling
**Why Needed:** Base paper handles isolated signs; we need continuous recognition

**Options:**

**A. Bidirectional LSTM (BiLSTM)**
- Forward + backward context
- Proven for sequential data
- Lower computational cost
- Good for real-time systems

**B. Transformer Encoder**
- Multi-head self-attention
- Better long-range dependencies
- Parallel processing (faster training)
- State-of-the-art for sequences

**Implementation:**
- Input: Fused features (T × D_fusion)
- Layers: 2-4 BiLSTM or 6-8 Transformer layers
- Output: Contextualized embeddings (T × D_model)

#### 5.2.2 CTC Alignment
**Why Needed:** Continuous signing has no frame-level annotations

**How CTC Works:**
1. Model outputs probability distribution per frame
2. CTC allows repetitions and blank tokens
3. Decoding collapses to gloss sequence
4. Training: Aligns predictions to ground truth glosses

**Advantages:**
- No need for frame-level labels
- Handles variable signing speeds
- Supports co-articulation

#### 5.2.3 Language Processing Pipeline
**Components:**

1. **Caption Buffering**
   - Accumulate gloss tokens in sliding window
   - Filter duplicates
   - Temporal ordering

2. **Gloss-to-Text Translation**
   - Seq2Seq model (LSTM or Transformer)
   - Input: ISL gloss sequence
   - Output: English sentence
   - Trained on parallel ISL-English corpus

3. **Grammar Correction**
   - Rule-based: Fix common ISL→English patterns
   - LM-based: T5/BART fine-tuned on grammar
   - Fluency enhancement

4. **Text-to-Speech**
   - Offline: gTTS, pyttsx3 (fast)
   - Cloud: Google/AWS/Azure TTS (natural)
   - Neural: Tacotron 2 (best quality)

---

## 6. Key Architectural Decisions

### 6.1 Why Multi-Feature Learning?

**Decision:** Use RGB + Pose dual-stream (from base paper)

**Justification:**
- RGB alone: Sensitive to lighting, clothing, background
- Pose alone: Sensitive to tracking errors, occlusions
- Combined: Robust to both failure modes
- Base paper shows +5-8% accuracy improvement

**Alternative Considered:** RGB-only with data augmentation  
**Why Rejected:** Lower accuracy, less robust to appearance variations

### 6.2 Why Attention-Based Fusion?

**Decision:** Adaptive attention weighting (from base paper)

**Justification:**
- Better than concatenation (+3-5% accuracy)
- Adapts to input quality (e.g., poor lighting)
- Interpretable (attention weights visualizable)

**Alternative Considered:** Simple concatenation + MLP  
**Why Rejected:** Fixed fusion, no adaptability

### 6.3 Why CTC Alignment?

**Decision:** Use CTC for continuous sign recognition

**Justification:**
- No frame-level annotations needed
- Handles variable signing speeds
- Proven for speech recognition (similar problem)

**Alternative Considered:** Sliding window detection  
**Why Rejected:** Requires segmentation, misses co-articulation

### 6.4 Why BiLSTM vs Transformer?

**Decision:** Support both, choose based on use case

**BiLSTM:**
- Pros: Lower latency, less memory, good for real-time
- Cons: Sequential processing, limited long-range

**Transformer:**
- Pros: Better accuracy, parallel training, SOTA
- Cons: Higher latency, more memory

**Recommendation:**
- **Real-time deployment:** BiLSTM
- **Offline/batch processing:** Transformer

### 6.5 Why Language Correction?

**Decision:** Add gloss-to-text translation + grammar correction

**Justification:**
- Raw gloss output is incomprehensible to non-signers
- ISL grammar differs from English (SOV vs SVO)
- Improves usability significantly

**Alternative Considered:** Output glosses only  
**Why Rejected:** Poor user experience

---

## 7. Architecture Advantages

### 7.1 Modularity
- Easy to replace individual components
- Independent testing and debugging
- Supports incremental development

### 7.2 Scalability
- Works with 100 to 2000+ glosses
- Adaptable to other sign languages
- Cloud or edge deployment

### 7.3 Research Alignment
- Based on peer-reviewed base paper
- Extends with state-of-the-art techniques
- Reproducible experiments

### 7.4 Real-Time Performance
- Optimized inference pipelines
- Asynchronous processing
- Target <500ms latency achievable

### 7.5 Robustness
- Multi-feature learning
- Attention-based adaptive fusion
- Handles variations in environment and signer

---

## 8. Architecture Limitations & Mitigations

### 8.1 Limitation: Requires Good Pose Estimation
**Mitigation:**
- Use robust pose estimators (MediaPipe Holistic)
- Fallback to RGB-only if pose fails
- Attention fusion reduces weight on poor pose data

### 8.2 Limitation: Limited ISL Training Data
**Mitigation:**
- Transfer learning from ASL (pretrain on large-scale data)
- Data augmentation (speed, rotation, cropping)
- Fine-tuning strategy (freeze-unfreeze)

### 8.3 Limitation: Real-Time Latency Constraints
**Mitigation:**
- Use efficient backbones (ResNet-18 vs ResNet-50)
- Model quantization (FP16/INT8)
- Asynchronous processing pipelines

### 8.4 Limitation: Grammar Correction Quality
**Mitigation:**
- Use pretrained LMs (T5, BART)
- Fine-tune on ISL-English parallel corpus
- Rule-based post-processing for common errors

---

## 9. Color Coding Convention (For Diagrams)

To clearly communicate architectural contributions:

| Color | Meaning | Examples |
|-------|---------|----------|
| 🟢 **Green** | Existing from literature | CNN backbone, Pose estimation |
| 🔵 **Blue** | Modified algorithms | Attention fusion (adapted from base paper) |
| 🔴 **Pink/Red** | Novel contributions | Caption buffering, Language correction |

**Usage:**
- In presentations: Use these colors consistently
- In papers: Explicitly label "from [base paper]" vs "our contribution"
- In code: Document module sources

---

## 10. Related Documents

**For detailed modular flows:** [03_complete_modular_flow.md](03_complete_modular_flow.md)  
**For algorithmic details:** [04_algorithmic_design.md](04_algorithmic_design.md)  
**For model architecture:** [05_model_architecture_details.md](05_model_architecture_details.md)  
**For implementation:** [DETAILED_WORKFLOW.md](DETAILED_WORKFLOW.md)  
**For presentation:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

**Document Version:** 1.0  
**Last Updated:** January 24, 2026  
**Purpose:** High-level system architecture overview
