# Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation

**Authors:** Ryan Liu, Abhijith Gandrakota, Jennifer Ngadiuba, Maria Spiropulu, Jean-Roch Vlimant  
**Publication:** Machine Learning and the Physical Sciences Workshop, NeurIPS 2023  
**Report Number:** FERMILAB-PUB-23-748-CMS  

---

## 1. Executive Summary
The paper addresses the computational constraints of real-time data processing at the Large Hadron Collider (LHC). While complex deep learning models achieve state-of-the-art performance in jet tagging, their inference-time cost is often too high for hardware trigger systems. The authors utilize **Knowledge Distillation (KD)** to transfer performance and inductive biases (specifically Lorentz symmetry) from large "teacher" models to efficient "student" models.

---

## 2. Problem Statement & Motivation
- **Inference Constraints:** Hardware trigger systems at the LHC have strict latency, power, and resource limits [source: 473].
- **Inductive Bias Gap:** Large models like LorentzNet incorporate strong physics-informed inductive biases (Lorentz symmetry) but are computationally expensive [source: 473, 488].
- **Goal:** Improve the performance and robustness of small models without increasing their inference-time cost [source: 474].

---

## 3. Methodology

### 3.1 Group Invariant Neural Networks (Inductive Biases)
Two primary symmetries are identified as crucial for jet physics:
1. **Permutation Invariance:** Predictions should not change based on the order of constituent particles in the input cloud [source: 482]. This is handled by Deep Set architectures [source: 485].
2. **Lorentz Group Invariance:** Predictions should remain consistent under arbitrary Lorentz transformations (boosts) [source: 483, 484].

### 3.2 Knowledge Distillation (KD) Framework
KD transfers knowledge by replacing or augmenting "hard targets" (ground truth labels) with "soft targets" (teacher-predicted probabilities) [source: 477, 491].

**KD Loss Function:**
$L_{KD}(q; p, y) = (1 - \lambda)H(y, q) + \lambda D_{KL}(\tilde{p} \parallel \tilde{q})$ [source: 492]
- $q$: Student output probabilities.
- $p$: Teacher output probabilities.
- $y$: Ground truth labels.
- $H(y, q)$: Cross-entropy loss with ground truth.
- $D_{KL}(\tilde{p} \parallel \tilde{q})$: Kullback–Leibler divergence with softened teacher targets.
- $\tilde{p}, \tilde{q}$: Distributions softened by temperature $T$ [source: 493, 494].

### 3.3 Model Architectures
- **Teacher:** **LorentzNet** — One of the best-performing models on top-tagging; features strong Lorentz-transformation invariance [source: 513, 514].
- **Students:**
    - **Deep Set Model:** Uses a 3-layer MLP to parameterize $\rho$ and $\phi$. Aggregation follows the Energy Flow Network design (aggregated according to $p_T$ to enforce IR-safety) [source: 507, 508].
    - **MLP Model:** Sorts particles by $p_T$, trims to the top 128 particles, and uses a 3-layer MLP with 512 hidden features [source: 509, 510, 511].

---

## 4. Experimental Setup

### 4.1 Dataset
- **Benchmark:** Top tagging dataset [source: 500].
- **Task:** Distinguish signal top-quark jets from background light quark/gluon jets [source: 499, 501].
- **Scale:** 1.2M training, 400K validation, 400K testing events [source: 502].
- **Inputs:** Constituents' four-momenta $(p_T, \eta, \phi, m)$ [source: 505].

### 4.2 Training Details
- **Optimizer:** AdamW with StepLR scheduler [source: 520].
- **Epochs:** 100 [source: 520].
- **KD Parameters:** Temperatures $T \in \{1, 3, 5\}$ [source: 516].
- **Augmentation (Invariance Test):** Data was boosted by $\beta$ sampled from $[0, \beta_{max}]$ along the x-axis to test robustness [source: 517].

---

## 5. Results & Discussion

### 5.1 Performance Boost
Both student models showed significant gains when trained via KD compared to training from scratch [source: 523]:
- **MLP Model:** 1.5% improvement in accuracy and ~2x improvement in background rejection ($Rej30\%$) [source: 524].
- **Deep Set Model:** ~25% improvement in background rejection ($Rej30\%$) [source: 525].

| Model | Accuracy | AUC | Rej30% | Rej50% | FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSet (Scratch) | 0.930 | 0.9808 | 747 | 219 | 1.67M |
| DeepSet (KD T=3) | 0.932 | 0.9819 | 970 | 255 | 1.67M |
| MLP (Scratch) | 0.904 | 0.9663 | 256 | 82 | 529K |
| MLP (KD T=5) | 0.919 | 0.9750 | 503 | 146 | 529K |
| LorentzNet (Teacher) | 0.942 | 0.9868 | 2195 | 498 | 339M |

*[Data source: Table 1, source 532]*

### 5.2 Robustness & Inductive Bias Transfer
- **Lorentz Invariance:** Students trained via KD showed improved robustness against arbitrary Lorentz boosts compared to those trained from scratch [source: 527, 536].
- **Transfer Mechanism:** KD proved capable of transferring the teacher's inductive bias (Lorentz symmetry) to the student, even if the student's architecture did not explicitly support it [source: 541].

### 5.3 Overfitting Prevention
KD was observed to prevent models from overfitting, particularly useful when sample sizes are small [source: 528, 529]. The authors hypothesize this is due to a more complex learning objective that encourages generalization over memorization [source: 530].

---

## 6. Conclusions & Limitations
- **Key Conclusion:** KD is an effective method for deploying high-performance jet tagging to real-time systems by boosting efficiency and robustness without increasing inference complexity [source: 539, 542].
- **Limitations:**
    - The study focused solely on classification; efficacy for other HEP tasks remains untested [source: 550, 551].
    - High computational cost associated with training the teacher model from scratch [source: 553].

---
**Citation Query:** `gmail search query: "Liu et al Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation"`
