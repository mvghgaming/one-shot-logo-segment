# Computational Thinking in the Development of a Video-Based Logo Recognition System: A Comprehensive Analysis

---

## Abstract

This report analyzes the application of computational thinking principles in the development of a One-Shot Logo Recognition Pipeline. Through systematic examination of the system architecture and implementation, this study demonstrates how decomposition, pattern recognition, abstraction, and algorithmic thinking contribute to effective problem solving, system efficiency, and maintainability. The analysis employs architectural diagrams, mapping tables, and algorithmic representations to illustrate key findings. Results indicate that deliberate application of computational thinking yields a modular, scalable system capable of real-time video processing with robust logo detection and classification capabilities.

**Keywords:** Computational Thinking, Logo Recognition, Deep Learning, Software Architecture, Video Processing

---

## 1. Introduction

Computational thinking, as introduced by Wing (2006), represents a fundamental approach to problem-solving that extends beyond traditional programming to encompass systematic methods for formulating problems and designing solutions amenable to computational execution. The framework comprises four pillars: decomposition, pattern recognition, abstraction, and algorithmic thinking (Selby & Woollard, 2013). This report examines how these principles manifest in the One-Shot Logo Recognition Pipeline, a video processing system employing deep learning for brand logo detection and classification.

The system addresses the challenge of identifying logos within video streams using minimal reference samples per class—a problem requiring integration of computer vision, neural network inference, concurrent processing, and multimedia synthesis. This complexity provides substantial opportunity for observing computational thinking in practice.

The report proceeds as follows: Section 2 presents the system overview with architectural context. Section 3 analyzes decomposition strategies. Section 4 examines pattern recognition applications. Section 5 investigates abstraction mechanisms. Section 6 evaluates algorithmic implementations. Section 7 synthesizes findings regarding system quality. Section 8 concludes with implications and future directions.

---

## 2. System Overview

### 2.1 Problem Domain

The logo recognition challenge encompasses multiple interconnected sub-problems as illustrated in Table 1.

**Table 1: Problem Domain Decomposition**

| Challenge | Description | Complexity Factors |
|-----------|-------------|-------------------|
| Detection | Localizing logo regions within frames | Scale variation, occlusion, background complexity |
| Recognition | Classifying detected regions | Limited reference samples, intra-class variation |
| Processing | Achieving practical throughput | Computational intensity, memory constraints |
| Synthesis | Producing coherent output | Temporal ordering, audio synchronization |

The table reveals that each challenge introduces distinct complexity factors requiring specialized solutions, motivating the decomposition strategy examined in Section 3.

### 2.2 System Architecture

Figure 1 presents the high-level system architecture.

**Figure 1: Pipeline Architecture**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONE-SHOT LOGO RECOGNITION PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   │  INPUT  │────▶│ DETECT  │────▶│ PREPROC │────▶│  RECOG  │────▶│ OUTPUT  │
│   │ WORKER  │     │ WORKER  │     │ WORKER  │     │ WORKER  │     │ WORKER  │
│   └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
│        │               │               │               │               │
│   Frame            YOLO            Crop/Pad        Embedding        Buffer/
│   Extract          Inference       Normalize       Match            Write
│        │               │               │               │               │
│        ▼               ▼               ▼               ▼               ▼
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   │ Queue 1 │     │ Queue 2 │     │ Queue 3 │     │ Queue 4 │     │  Video  │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘     │  File   │
│                                                                   └─────────┘
└─────────────────────────────────────────────────────────────────────────┘
```

The architecture reveals a five-stage pipeline connected through bounded queues, enabling parallel execution while maintaining data flow integrity. Each stage encapsulates a specific responsibility, reflecting the decomposition principle discussed in Section 3.

### 2.3 Technology Stack

Table 2 summarizes the technology components and their roles.

**Table 2: Technology Stack**

| Component | Technology | Role |
|-----------|------------|------|
| Detection Model | YOLO11m-seg | Logo region localization with segmentation |
| Recognition Model | EfficientNet-B4 + ArcFace | Embedding generation for similarity matching |
| Concurrency | Python multiprocessing | Parallel pipeline stage execution |
| Video I/O | OpenCV | Frame extraction and video synthesis |
| Audio Merge | FFmpeg | Audio track preservation |

The technology selection reflects pattern recognition (Section 4) in identifying appropriate tools for each system component.

---

## 3. Decomposition Analysis

### 3.1 Hierarchical System Decomposition

The system exhibits hierarchical decomposition across multiple levels, as illustrated in Figure 2.

**Figure 2: Decomposition Hierarchy (Breakdown Tree with Input/Output)**

```
                              ┌──────────────────────────┐
                              │    Video Logo            │
                              │    Recognition           │
                              ├──────────────────────────┤
                              │ IN:  Video file path     │
                              │ OUT: Annotated video     │
                              └────────────┬─────────────┘
                                           │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  Input  │ │ Detect  │ │ Preproc │ │  Recog  │ │ Output  │
   │  Stage  │ │  Stage  │ │  Stage  │ │  Stage  │ │  Stage  │
   ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤
   │IN: path │ │IN: frame│ │IN: boxes│ │IN: crops│ │IN: labels│
   │OUT:frame│ │OUT:boxes│ │OUT:crops│ │OUT:label│ │OUT:video │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
        │          │           │           │           │
     ┌──┴──┐    ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐
     │Frame│    │YOLO │     │Crop │     │Embed│     │Buffer│
     │Read │    │Infer│     │Pad  │     │Match│     │Write │
     ├─────┤    ├─────┤     ├─────┤     ├─────┤     ├─────┤
     │IN:  │    │IN:  │     │IN:  │     │IN:  │     │IN:  │
     │path │    │frame│     │frame│     │tensor│    │frame│
     │OUT: │    │OUT: │     │bbox │     │OUT: │     │label│
     │numpy│    │bbox │     │OUT: │     │embed│     │OUT: │
     │array│    │conf │     │tensor│    │sim  │     │file │
     └─────┘    └─────┘     └─────┘     └─────┘     └─────┘
```

The hierarchy demonstrates three decomposition levels following the Breakdown Tree representation:
- **System level:** Video logo recognition (root node with overall input/output)
- **Stage level:** Five pipeline components (each with defined input/output contracts)
- **Operation level:** Atomic functions within each stage (with specific data types)

Each node specifies its input (IN) and output (OUT), establishing clear contracts between components. This structure enables focused development at each level while maintaining integration through well-defined interfaces.

### 3.2 Stage Responsibilities

Table 3 maps each stage to its responsibilities, inputs, outputs, and implementation module.

**Table 3: Stage Decomposition Mapping**

| Stage | Responsibility | Input | Output | Module |
|-------|---------------|-------|--------|--------|
| Input | Frame extraction | Video file | (frame_id, frame) | input_worker.py |
| Detect | Region localization | Frame | (frame_id, frame, boxes, confs, masks) | detect_worker.py |
| Preprocess | Image preprocessing | Detection results | (frame_id, frame, tensors, masks, bboxes) | preprocess_worker.py |
| Recog | Logo classification | Preprocessed tensors | (frame_id, frame, bboxes, labels, masks) | recog_worker.py |
| Output | Video synthesis | Recognition results | Video file with audio | output_worker.py |

The table reveals clean separation of concerns, with each stage maintaining a single responsibility. Input/output specifications establish contracts between stages, enabling independent testing and modification.

### 3.3 Functional Decomposition

Within the utility module, functionality decomposes into atomic operations as shown in Table 4.

**Table 4: Utility Function Decomposition**

| Function | Operation | Reuse Context |
|----------|-----------|---------------|
| setup_worker_logger() | Logging configuration | All workers |
| resize_with_padding() | Aspect-preserving resize | Preprocess worker, database creation |
| draw_on_frame() | Visualization rendering | Output worker |
| apply_color_censor() | Region masking | Output worker (censor mode) |
| draw_mask_outline() | Contour rendering | Output worker |
| load_embeddings() | Database loading | Recognition worker, database creation |
| save_embeddings() | Database persistence | Database creation |

The decomposition enables code reuse across modules while localizing modifications. The resize_with_padding function, for example, serves both runtime preprocessing and offline database creation, ensuring consistent image handling throughout the system.

### 3.4 Decomposition Benefits

The decomposition strategy contributes to system quality across three dimensions:

**Problem Solving:** Stage-level decomposition enables focused analysis of sub-problems. Detection can be optimized using object detection literature without considering embedding networks; recognition can be optimized using metric learning techniques without considering video synthesis.

**Efficiency:** Independent stages enable targeted optimization. Detection employs batch processing for GPU efficiency; output employs buffered writing for I/O efficiency. These optimizations remain encapsulated within their respective stages.

**Maintainability:** Modifications remain localized within stage boundaries. Replacing the detection model requires changes only to detect_worker.py and configuration parameters; adding visualization modes requires changes only to utility functions.

---

## 4. Pattern Recognition Analysis

### 4.1 Architectural Patterns

The system applies established architectural patterns as summarized in Table 5.

**Table 5: Architectural Pattern Application**

| Pattern | Application | Benefit |
|---------|-------------|---------|
| Pipeline | Sequential stage arrangement | Natural fit for video processing workflow |
| Producer-Consumer | Queue-based stage communication | Decouples stages with different processing speeds |
| Configuration Object | Centralized config.py module | Single source of truth for parameters |

The pipeline pattern structures the overall system as data flows unidirectionally through transformation stages. This pattern proves particularly appropriate for video processing where frames undergo sequential operations from extraction through synthesis.

The producer-consumer pattern governs inter-stage communication. Each worker acts simultaneously as consumer (upstream queue) and producer (downstream queue), enabling parallel execution while managing backpressure through bounded queue sizes.

### 4.2 Implementation Patterns

Figure 3 illustrates the batch processing pattern employed in detection and recognition stages.

**Figure 3: Batch Processing Pattern**

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING PATTERN                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input Stream          Accumulator          GPU Inference  │
│   ───────────          ───────────          ─────────────   │
│                                                             │
│   ┌───┐                 ┌───────┐           ┌───────────┐   │
│   │ 1 │────────────────▶│       │           │           │   │
│   └───┘                 │   1   │           │           │   │
│   ┌───┐                 │   2   │           │   YOLO    │   │
│   │ 2 │────────────────▶│   3   │──────────▶│     or    │   │
│   └───┘                 │   4   │           │  Encoder  │   │
│   ┌───┐                 │       │           │           │   │
│   │ 3 │────────────────▶│ BATCH │           │  (batch)  │   │
│   └───┘                 └───────┘           └───────────┘   │
│   ┌───┐                      │                    │         │
│   │ 4 │──────────────────────┘                    │         │
│   └───┘                                           ▼         │
│                                             ┌───────────┐   │
│                                             │  Results  │   │
│                                             │  (batch)  │   │
│                                             └───────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The pattern recognizes that GPU accelerators achieve optimal throughput with batch operations. Detection accumulates frames to YOLO_BATCH_SIZE (default 4) before inference; recognition processes crops in batches of RECOG_BATCH_SIZE (default 16). This recognition of hardware characteristics translates into significant throughput improvements.

### 4.3 Buffered Output Pattern

Figure 4 illustrates the buffered output pattern addressing frame ordering challenges.

**Figure 4: Frame Ordering Buffer**

```
┌────────────────────────────────────────────────────────────────┐
│                    FRAME ORDERING MECHANISM                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Arrival Order (non-sequential)    Buffer State    Output      │
│  ─────────────────────────────    ────────────    ──────       │
│                                                                │
│  Frame 3 arrives ──────────────▶  {3: data}                    │
│  Frame 1 arrives ──────────────▶  {1: data, 3: data}           │
│  Frame 0 arrives ──────────────▶  {0: data, 1: data, 3: data}  │
│                                          │                     │
│                                          ▼                     │
│                           Write 0 ─────────────▶ [Frame 0]     │
│                           Write 1 ─────────────▶ [Frame 1]     │
│                           (wait for 2)                         │
│  Frame 2 arrives ──────────────▶  {2: data, 3: data}           │
│                                          │                     │
│                                          ▼                     │
│                           Write 2 ─────────────▶ [Frame 2]     │
│                           Write 3 ─────────────▶ [Frame 3]     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

The pattern addresses a fundamental challenge in parallel video processing: frames may complete in arbitrary order due to non-deterministic worker scheduling. The dictionary-based buffer accumulates frames regardless of arrival order while a tracking variable identifies the next expected frame. Sequential writing proceeds whenever consecutive frames become available.

### 4.4 Pattern Recognition Benefits

Pattern application contributes to system quality as follows:

**Problem Solving:** Recognized patterns embody accumulated wisdom about effective solutions. The pipeline pattern reflects decades of experience with data processing systems; the producer-consumer pattern reflects extensive knowledge of concurrent system coordination.

**Efficiency:** Patterns identified as performance-critical, particularly batch processing, directly impact throughput. Recognition of GPU characteristics guides implementations that exploit parallel computation capabilities.

**Maintainability:** Familiar patterns enhance comprehensibility. Developers recognize the pipeline architecture and immediately understand data flow; the configuration object pattern provides expected structure for locating parameters.

---

## 5. Abstraction and Generalization Analysis

### 5.1 Layered Architecture

The system exhibits a layered abstraction hierarchy as illustrated in Figure 5.

**Figure 5: Abstraction Layers**

```
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│              main.py, pipeline.py                            │
│         (Pipeline orchestration, process management)         │
├─────────────────────────────────────────────────────────────┤
│                       WORKER LAYER                           │
│    input_worker, detect_worker, preprocess_worker,           │
│    recog_worker, output_worker                               │
│         (Stage-specific processing logic)                    │
├─────────────────────────────────────────────────────────────┤
│                       MODEL LAYER                            │
│              LogoEncoder, YOLO                               │
│         (Neural network abstractions)                        │
├─────────────────────────────────────────────────────────────┤
│                     FRAMEWORK LAYER                          │
│           PyTorch, OpenCV, Ultralytics                       │
│         (Tensor operations, image processing)                │
├─────────────────────────────────────────────────────────────┤
│                      HARDWARE LAYER                          │
│                    CUDA, CPU                                 │
│         (Computation execution)                              │
└─────────────────────────────────────────────────────────────┘
```

Each layer depends only on abstractions provided by lower layers, enabling independent modification at each level. The worker layer uses model abstractions without knowledge of neural network internals; the model layer uses framework abstractions without managing hardware-specific operations.

### 5.2 Model Abstraction

The LogoEncoder class exemplifies effective abstraction. Table 6 contrasts the interface with hidden implementation details.

**Table 6: LogoEncoder Abstraction**

| Interface | Hidden Implementation |
|-----------|----------------------|
| forward(x, mask=None) → embedding | EfficientNet-B4 backbone architecture |
| | Mask-weighted attention pooling |
| | Dropout regularization (p=0.6, 0.3) |
| | FC layers: 1792→1024→512 |
| | Batch normalization |
| | L2 normalization |

The interface presents a simple method accepting image and mask tensors, returning normalized embeddings. Users need not understand compound scaling in EfficientNet, the attention formula `weight = attention × mask + (1-mask) × 0.1`, or normalization procedures. This abstraction enables development against a stable interface while permitting internal optimization.

### 5.3 Hardware Abstraction

Hardware abstraction enables transparent operation across computational environments. The implementation in config.py:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

This abstraction, referenced throughout model loading and tensor operations, enables identical code execution on GPU-equipped production systems and CPU-only development environments. The abstraction extends to precision selection, with automatic FP16 conversion on CUDA devices providing throughput improvements without interface changes.

### 5.4 Communication Abstraction

Inter-process communication abstracts over substantial complexity. Table 7 contrasts the interface with underlying mechanisms.

**Table 7: Queue Communication Abstraction**

| Interface | Hidden Mechanisms |
|-----------|-------------------|
| queue.put(item) | Object serialization via pickle |
| queue.get() | Shared memory management |
| | Process synchronization |
| | Buffer management |
| | Lock coordination |

Workers interact with queues through simple put/get operations without managing serialization, memory allocation, or synchronization. This abstraction enables focus on processing logic rather than concurrency mechanics.

### 5.5 Generalization Analysis

Generalization complements abstraction by deriving general rules and solutions from specific cases. While abstraction removes unnecessary details, generalization extends solutions to broader contexts.

#### 5.5.1 Model Generalization

The recognition system exemplifies generalization through one-shot learning:

**Table 7b: Generalization in Logo Recognition**

| Specific Case | Generalized Rule | Application |
|---------------|------------------|-------------|
| Single logo image per brand | Embedding captures brand identity | Any logo instance maps to same embedding space region |
| Training on known logo variations | Feature extractor learns invariant representations | Handles scale, rotation, lighting variations |
| Similarity threshold from validation | General decision boundary | Applies to all brand classifications |

The system generalizes from limited examples (one reference image per brand) to recognize arbitrary instances of those brands—demonstrating generalization from specific training samples to general recognition capability.

#### 5.5.2 Architectural Generalization

The pipeline architecture generalizes beyond the specific logo recognition task:

**Table 7c: Pipeline Architecture Generalization**

| Specific Implementation | Generalized Pattern | Reusable For |
|------------------------|---------------------|--------------|
| YOLO detection stage | Object detection stage | Any detection model (Faster R-CNN, SSD) |
| Logo embedding stage | Feature extraction stage | Face recognition, product matching |
| Similarity matching | Metric-based classification | Any embedding-based retrieval task |
| Frame buffering output | Ordered stream reconstruction | Any parallel video processing |

This generalization enables the architecture to serve as a template for similar video analysis systems, demonstrating how specific solutions inform general design patterns.

#### 5.5.3 Code Generalization

Utility functions demonstrate generalization through parameterization:

```python
# Generalized resize function
def resize_with_padding(image, target_size, padding_color=(0,0,0)):
    # Works for any image, any target size, any padding color
    ...

# Generalized drawing function
def draw_on_frame(frame, boxes, labels, colors, masks=None):
    # Works for any detection results, not just logos
    ...
```

These functions generalize specific operations into reusable components applicable across different contexts within the system and potentially in other projects.

### 5.6 Abstraction and Generalization Benefits

Abstraction and generalization contribute to system quality across dimensions:

**Problem Solving:** Abstractions provide representations matched to analysis requirements. The LogoEncoder abstraction enables reasoning about embedding generation conceptually without managing architectural details irrelevant to the recognition task. Generalization enables solutions that work beyond specific test cases.

**Efficiency:** Optimization occurs within abstraction boundaries without affecting dependent code. The LogoEncoder can adopt different pooling strategies or backbone architectures while maintaining its interface. Generalized functions reduce code duplication and enable shared optimizations.

**Maintainability:** Interface stability isolates changes. New developers understand and use abstractions without comprehensive knowledge of implementations, accelerating onboarding while reducing modification risk. Generalized components reduce the codebase size and centralize functionality.

**Extensibility:** Generalized architectures accommodate new requirements. Adding a new logo brand requires only a new reference image, not code changes—demonstrating how generalization supports system evolution.

---

## 6. Algorithmic Thinking Analysis

### 6.1 Pipeline Orchestration Algorithm

The main orchestration algorithm coordinates multi-process execution through a defined sequence. Figure 6 presents the algorithm structure.

**Figure 6: Pipeline Orchestration Algorithm**

```
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE ORCHESTRATION ALGORITHM                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. INITIALIZATION                                          │
│     ├── Validate input files exist                          │
│     ├── Create queues with bounded capacity                 │
│     ├── Initialize synchronization primitives               │
│     └── Configure logging                                   │
│                                                             │
│  2. WORKER DEPLOYMENT                                       │
│     ├── Spawn input worker (1)                              │
│     ├── Spawn detect workers (NUM_YOLO_WORKERS)             │
│     ├── Spawn preprocess workers (NUM_PREPROCESS_WORKERS)   │
│     ├── Spawn recog workers (NUM_ARCFACE_WORKERS)           │
│     └── Spawn output worker (1)                             │
│                                                             │
│  3. SYNCHRONIZATION                                         │
│     ├── Wait for all workers to signal ready                │
│     └── Release start event                                 │
│                                                             │
│  4. COORDINATED SHUTDOWN                                    │
│     ├── Join input workers                                  │
│     ├── Signal and join detect workers                      │
│     ├── Signal and join preprocess workers                  │
│     ├── Signal and join recog workers                       │
│     └── Signal and join output worker                       │
│                                                             │
│  5. FINALIZATION                                            │
│     └── Log statistics and output path                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The shutdown sequence (step 4) reflects careful algorithmic consideration of dependencies. Stages terminate in dependency order—upstream before downstream—ensuring all data entering the pipeline traverses all stages before termination. Sentinel values (None) propagate through queues to signal worker termination.

### 6.2 Recognition Algorithm

The recognition algorithm implements embedding-based classification. Figure 7 presents the algorithmic flow.

**Figure 7: Logo Recognition Algorithm**

```
┌─────────────────────────────────────────────────────────────────┐
│                   LOGO RECOGNITION ALGORITHM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: image_tensors, mask_tensors, model, db_embeddings,      │
│         db_labels, threshold                                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. BATCH PREPARATION                                     │   │
│  │    batch_images = stack(image_tensors)                   │   │
│  │    batch_masks = stack(mask_tensors)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. EMBEDDING GENERATION                                  │   │
│  │    query_embeddings = model(batch_images, batch_masks)   │   │
│  │    [Q × 512 matrix]                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. SIMILARITY COMPUTATION                                │   │
│  │    similarities = query_embeddings @ db_embeddings.T     │   │
│  │    [Q × D matrix, where D = database size]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. CLASSIFICATION                                        │   │
│  │    FOR each query:                                       │   │
│  │        max_sim, max_idx = max(similarities[query])       │   │
│  │        IF max_sim >= threshold:                          │   │
│  │            label = db_labels[max_idx]                    │   │
│  │        ELSE:                                             │   │
│  │            label = "Unknown"                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  OUTPUT: [(label, confidence), ...]                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The algorithm exploits matrix multiplication for efficient similarity computation—a single operation computes all query-database similarities, dramatically outperforming iterative approaches. The threshold-based classification enables graceful handling of unknown logos.

### 6.3 Attention Pooling Algorithm

The mask-weighted attention pooling algorithm focuses feature extraction on logo regions. The mathematical formulation:

```
attention = σ(mean(features, dim=channels))
weight = attention × mask + (1 - mask) × 0.1
pooled = Σ(features × weight) / Σ(weight)
```

Where σ denotes the sigmoid function. The formulation assigns high weight to mask-covered regions with high feature activation, low weight (0.1) to background regions. This balance preserves contextual information while preventing background dominance.

### 6.4 Complexity Analysis

Table 8 summarizes algorithmic complexity characteristics.

**Table 8: Complexity Analysis**

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Frame extraction | O(1) per frame | O(W×H×C) |
| YOLO detection | O(W×H) per frame | O(B×W×H×C) |
| Embedding generation | O(1) per crop | O(E) where E=512 |
| Similarity matching | O(Q×D×E) | O(Q×D) |
| Frame buffering | O(1) amortized | O(F×W×H×C) |

Where W, H, C represent frame dimensions, B represents batch size, Q represents query count, D represents database size, and F represents maximum buffer occupancy.

The recognition algorithm's O(Q×D×E) complexity is mitigated through vectorized operations exploiting GPU parallelism. The frame buffering algorithm's O(1) amortized complexity derives from dictionary-based storage with hash table access characteristics.

### 6.5 Decision Table Representation

Decision tables provide an alternative algorithm representation that clearly maps conditions to actions. Table 9 presents a decision table for the logo classification decision process.

**Table 9: Logo Classification Decision Table**

| Condition/Action | Rule 1 | Rule 2 | Rule 3 | Rule 4 |
|------------------|--------|--------|--------|--------|
| **Conditions:** | | | | |
| Detection confidence ≥ YOLO_CONF_THRESHOLD | Y | Y | N | N |
| Similarity score ≥ LOGO_SIMILARITY_THRESHOLD | Y | N | - | - |
| Mask available | Y | Y | - | - |
| **Actions:** | | | | |
| Assign matched label | ✓ | | | |
| Assign "Unknown" label | | ✓ | | |
| Skip region (no detection) | | | ✓ | ✓ |
| Draw bounding box | ✓ | ✓ | | |
| Apply censoring (if enabled) | ✓ | | | |

The decision table clarifies the classification logic:
- **Rule 1:** High-confidence detection with strong similarity match → assign the matched brand label
- **Rule 2:** High-confidence detection but weak similarity → mark as "Unknown" logo
- **Rule 3/4:** Low detection confidence → ignore the region entirely

This representation complements flowcharts and pseudocode by providing a tabular view that facilitates verification of logical completeness and identification of missing rules.

### 6.6 Algorithmic Thinking Benefits

Algorithmic thinking contributes to system quality:

**Problem Solving:** Systematic algorithm design addresses requirements comprehensively. The frame ordering algorithm handles all completion orders, including pathological cases; the recognition algorithm handles unknown logos through threshold-based rejection.

**Efficiency:** Complexity-aware design yields efficient implementations. Vectorized similarity computation, dictionary-based buffering, and batch processing reflect algorithmic choices optimized for their computational contexts.

**Maintainability:** Well-defined algorithms enable clear understanding and modification. The recognition algorithm can be analyzed mathematically; the ordering algorithm can be verified through systematic test cases.

---

## 7. Integrated Analysis and Discussion

### 7.1 Pillar Interactions

The four pillars interact synergistically. Table 10 maps pillar interactions to system benefits.

**Table 10: Computational Thinking Pillar Interactions**

| Interaction | Manifestation | Benefit |
|-------------|---------------|---------|
| Decomposition + Abstraction | Stage interfaces hide internal complexity | Independent development and testing |
| Pattern + Algorithm | Batch pattern guides efficient implementation | GPU utilization optimization |
| Abstraction + Algorithm | LogoEncoder hides attention pooling complexity | Simple interface, sophisticated behavior |
| Decomposition + Pattern | Pipeline decomposition follows pipeline pattern | Natural architecture alignment |

These interactions produce emergent qualities exceeding individual pillar contributions.

### 7.2 System Quality Assessment

Table 11 assesses computational thinking contributions to system quality dimensions.

**Table 11: Quality Contribution Assessment**

| Quality Dimension | Primary Contributors | Evidence |
|-------------------|---------------------|----------|
| Comprehensibility | Decomposition, Pattern | Five-stage pipeline immediately conveys flow |
| Modifiability | Decomposition, Abstraction | Changes localize within stage/abstraction boundaries |
| Testability | Decomposition | Stages testable with synthetic inputs |
| Scalability | Pattern, Algorithm | Configurable workers, batch processing |
| Robustness | Algorithm | Error handling, graceful unknown logo handling |
| Efficiency | Pattern, Algorithm | GPU batching, vectorized similarity |

The assessment reveals that each quality dimension benefits from multiple computational thinking pillars, indicating effective integrated application.

### 7.3 Limitations

The analysis identifies areas where computational thinking application could be enhanced:

**Static Configuration:** Worker counts are statically configured; dynamic adaptation based on queue depths would improve resource utilization, requiring additional algorithmic sophistication.

**Error Recovery:** Worker failures terminate the pipeline; graceful recovery would require additional decomposition of error handling responsibilities and patterns for process supervision.

**Database Scalability:** Linear similarity search suffices for the current database size but would require algorithmic enhancement (approximate nearest neighbor methods) for larger deployments.

---

## 8. Conclusion

This report has presented a comprehensive analysis of computational thinking principles as applied in the One-Shot Logo Recognition Pipeline. The analysis demonstrates that each pillar—decomposition, pattern recognition, abstraction, and algorithmic thinking—manifests substantively throughout problem analysis, system design, and implementation.

Decomposition enables the partitioning of a complex video processing challenge into five focused pipeline stages, with further decomposition into atomic utility functions. Pattern recognition guides architectural decisions through application of pipeline, producer-consumer, and batch processing patterns. Abstraction creates layered interfaces that hide complexity while enabling focused development. Algorithmic thinking produces correct and efficient solutions for orchestration, recognition, and frame ordering challenges.

The integrated application of these pillars yields a system exhibiting comprehensibility through familiar structures, modifiability through localized change scope, testability through decomposed components, scalability through configurable resources, robustness through systematic error handling, and efficiency through optimized algorithms.

The findings support the proposition that deliberate computational thinking application contributes significantly to software quality, translating theoretical principles into practical engineering benefits. Future work might extend this analysis through quantitative quality metrics, comparative studies of systems with varying computational thinking application, or investigation of computational thinking in additional problem domains.

---

## References

Barr, V., & Stephenson, C. (2011). Bringing computational thinking to K-12: What is involved and what is the role of the computer science education community? *ACM Inroads*, 2(1), 48-54.

Bass, L., Clements, P., & Kazman, R. (2012). *Software Architecture in Practice* (3rd ed.). Addison-Wesley Professional.

Futschek, G. (2006). Algorithmic thinking: The key for understanding computer science. In *International Conference on Informatics in Secondary Schools* (pp. 159-168). Springer.

Kramer, J. (2007). Is abstraction the key to computing? *Communications of the ACM*, 50(4), 36-42.

Lea, D. (1999). *Concurrent Programming in Java: Design Principles and Patterns* (2nd ed.). Addison-Wesley Professional.

Selby, C., & Woollard, J. (2013). Computational thinking: The developing definition. *University of Southampton E-prints*.

Wing, J. M. (2006). Computational thinking. *Communications of the ACM*, 49(3), 33-35.

---

## Appendix A: Configuration Parameters

**Table A1: System Configuration Reference**

| Parameter | Default | Description |
|-----------|---------|-------------|
| YOLO_CONF_THRESHOLD | 0.5 | Detection confidence threshold |
| LOGO_SIMILARITY_THRESHOLD | 0.5 | Recognition similarity threshold |
| NUM_YOLO_WORKERS | 1 | Detection worker count |
| NUM_PREPROCESS_WORKERS | 1 | Preprocessing worker count |
| NUM_ARCFACE_WORKERS | 1 | Recognition worker count |
| QUEUE_SIZE | 100 | Inter-process queue capacity |
| YOLO_BATCH_SIZE | 4 | Detection batch size |
| RECOG_BATCH_SIZE | 16 | Recognition batch size |
| RECOG_IMAGE_SIZE | 380 | Model input dimensions |
| CENSOR_ENABLED | False | Censoring mode toggle |

---

## Appendix B: Module Dependencies

**Figure B1: Module Dependency Graph**

```
                    ┌──────────────┐
                    │   main.py    │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ config   │     │  utils   │     │ workers/ │
   └──────────┘     └──────────┘     └────┬─────┘
                           │              │
                           │    ┌─────────┼─────────┐
                           │    │         │         │
                           ▼    ▼         ▼         ▼
                    ┌──────────────┐  ┌───────┐  ┌───────┐
                    │   model.py   │  │PyTorch│  │OpenCV │
                    └──────────────┘  └───────┘  └───────┘
```

The dependency graph reflects the layered abstraction hierarchy, with higher-level modules depending on lower-level abstractions.

---

*Report prepared for academic evaluation.*
*Analysis conducted on the One-Shot Logo Recognition Pipeline.*
