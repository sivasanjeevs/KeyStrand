# Keystrand 🧬

> **Keystroke genomics & forensic hijack detection toolkit**

A lightweight bio-computational forensics engine that applies bioinformatics sequence alignment (Gotoh affine gaps), PSSM profiling, and Hidden Markov Models (Baum-Welch & Viterbi) to detect account takeovers and typing impostors.

---

## 🏗️ Architecture & Bioinformatics Pipeline

```
Typed Message → Dictionary Tokenization & Mapping
             → Pairwise Needleman-Wunsch / Smith-Waterman with Gotoh Affine Gaps
             → Edit-Operation String (M / S / I / D)
             → 3-State Baum-Welch HMM (Forward Log-Likelihood & Viterbi State Decoding)
             + Position-Specific Scoring Matrix (PSSM) Log-Odds Profile
             → Likelihood-Ratio Forensic Verdict Engine: Person_A / Person_B / Impostor
```

---

## 📁 Repository Modules

- **`data_gen.py` (Module 1: Calibration & Chat Generator)**:
  - Generates `calibration.csv` (2,400 labeled typing keystroke sequences for `Person_A`, `Person_B`, and `Impostor` across 40 interrogation words in `careful` and `rushing`/`sleepy` states).
  - Generates `chat_history.json` with 10 realistic messages, including the 3 AM suspicious ransom demand sent from a hijacked session.

- **`aligner.py` (Module 2: The Forensic Aligner)**:
  - **Needleman-Wunsch (Global)** & **Smith-Waterman (Local)** implemented from scratch with **Gotoh Affine Gap Penalties** ($M, I_x, I_y$ 3-matrix DP).
  - **Empirical Log-Odds Scoring Matrix**: $S(a, b) = \log_2 \frac{q_{ab}}{p_a p_b}$ learned directly from calibration keystrokes, plus physical QWERTY Euclidean distance matrix.
  - **Gumbel Statistical Significance**: Permutation test (100 shuffles) with `scipy.stats.gumbel_r` fitting to report empirical and theoretical p-values.
  - **Edit-Op Extractor**: Maps pairwise alignments to `M` (Match), `S` (Substitution), `I` (Insertion/key repeat), `D` (Deletion/omission).

- **`profiler.py` (Module 3: The Profiler)**:
  - **PSSM Generation**: Position-Specific Scoring Matrix for variant spellings of words.
  - **Baum-Welch HMM (`hmmlearn`)**: 3 hidden states (`Careful`, `Rushing`, `Sleepy`) emitting 4 observation categories (`M`, `S`, `I`, `D`).
  - **Forward Algorithm**: Computes likelihood $\log P(O \mid \lambda)$ under each suspect's model.
  - **Viterbi Algorithm**: Decodes the hidden mood trajectory across the message.

- **`engine.py` (Module 4: Evaluation & Verdict Engine)**:
  - Tokenizes messages, handles chat contractions, collapses fatigue repeats, and maps tokens with `pyspellchecker`.
  - Multi-factor Likelihood-Ratio decision engine with confidence scoring.
  - **4-Model Benchmark**: Evaluates and computes 3x3 confusion matrices for:
    1. Basic Edit Distance Only
    2. Scoring Matrix Only
    3. HMM Only
    4. Combined Forensic Toolkit

- **`app.py` (Module 5: Streamlit Forensic Dashboard)**:
  - Cyber-forensics investigation dashboard with interactive chat bubble view.
  - Viterbi mood decode visualization colored by hidden state.
  - Seaborn dot matrix heatmap and HMM likelihood comparisons.
  - 3x3 confusion matrix benchmark suite and interactive forensic sandbox.

- **`verify.py`**:
  - Test suite verifying mathematical correctness of all algorithms without hallucinations.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Run Verification Tests
```bash
python data_gen.py
python verify.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

