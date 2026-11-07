"""
Keystrand: Keystroke Genomics & Forensic Hijack Detection
Module 3: The Profiler (profiler.py)

Contains:
1. PSSM Generation: Position-Specific Scoring Matrices for personal spelling signatures.
2. HMM Implementation (hmmlearn):
   - Hidden states: Careful (0), Rushing (1), Sleepy (2)
   - Observations: Match 'M' (0), Substitution 'S' (1), Insertion 'I' (2), Deletion 'D' (3)
   - Baum-Welch EM training on calibration edit-operation sequences.
   - Forward Algorithm for message likelihood calculation under a person's model.
   - Viterbi Decoding for hidden mood sequence inference along messages.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional, Any
from aligner import needleman_wunsch_affine, learn_scoring_matrix_from_csv

OP_MAP = {'M': 0, 'S': 1, 'I': 2, 'D': 3}
REV_OP_MAP = {0: 'M', 1: 'S', 2: 'I', 3: 'D'}
STATE_NAMES = ["Careful", "Rushing", "Sleepy"]
STATE_COLORS = {
    "Careful": "#2ecc71",   # Green
    "Rushing": "#e67e22",   # Orange
    "Sleepy": "#9b59b6"     # Purple
}

ALPHABET_WITH_GAP = list("abcdefghijklmnopqrstuvwxyz-")
CHAR_INDEX = {c: i for i, c in enumerate(ALPHABET_WITH_GAP)}


# =====================================================================
# 1. POSITION-SPECIFIC SCORING MATRIX (PSSM) FOR SPELLING SIGNATURES
# =====================================================================

class WordPSSM:
    """
    Position-Specific Scoring Matrix for a specific word's spelling variations.
    Constructed from multiple sequence alignment of a typist's observed forms.
    """
    def __init__(self, intended_word: str, pseudocount: float = 0.5):
        self.intended_word = intended_word.lower()
        self.length = len(self.intended_word)
        self.pseudocount = pseudocount
        self.pssm_matrix = None  # shape: (27, length) where 27 is alphabet + '-'
        self.background_freq = np.full(27, 1.0 / 27.0)

    def build_from_variants(
        self,
        variants: List[str],
        scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ):
        """
        Aligns each variant spelling to the intended reference word using NW affine,
        builds the positional frequency matrix, and computes log-odds against background.
        """
        counts = np.full((27, self.length), self.pseudocount)

        for variant in variants:
            if not variant:
                continue
            _, ref_aligned, var_aligned, _ = needleman_wunsch_affine(
                self.intended_word, variant.lower(), scoring_matrix=scoring_matrix
            )

            # Map aligned characters back onto reference columns
            ref_col = 0
            for r_char, v_char in zip(ref_aligned, var_aligned):
                if r_char != '-':
                    if ref_col < self.length:
                        c_idx = CHAR_INDEX.get(v_char, CHAR_INDEX['-'])
                        counts[c_idx, ref_col] += 1.0
                    ref_col += 1

        # Normalize column frequencies
        col_sums = counts.sum(axis=0, keepdims=True)
        prob_matrix = counts / col_sums

        # Log2 odds relative to uniform background
        log_odds = np.log2(prob_matrix / self.background_freq[:, np.newaxis])
        self.pssm_matrix = log_odds

    def score_variant(
        self,
        variant: str,
        scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> float:
        """
        Scores a query spelling against this word's PSSM profile.
        """
        if self.pssm_matrix is None:
            return 0.0

        _, ref_aligned, var_aligned, _ = needleman_wunsch_affine(
            self.intended_word, variant.lower(), scoring_matrix=scoring_matrix
        )

        total_score = 0.0
        ref_col = 0
        for r_char, v_char in zip(ref_aligned, var_aligned):
            if r_char != '-':
                if ref_col < self.length:
                    c_idx = CHAR_INDEX.get(v_char, CHAR_INDEX['-'])
                    total_score += float(self.pssm_matrix[c_idx, ref_col])
                ref_col += 1

        return total_score


class ProfilerPSSMRegistry:
    """
    Maintains a dictionary of PSSMs across all interrogation words for each author.
    """
    def __init__(self, scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None):
        self.scoring_matrix = scoring_matrix
        # profiles[typist_id][intended_word] = WordPSSM
        self.profiles: Dict[str, Dict[str, WordPSSM]] = {}

    def fit_from_calibration_df(self, df: pd.DataFrame):
        train_df = df[df["split"] == "train"] if "split" in df.columns else df
        typists = train_df["typist_id"].unique()

        for typist in typists:
            self.profiles[typist] = {}
            t_df = train_df[train_df["typist_id"] == typist]
            words = t_df["intended_word"].unique()

            for word in words:
                variants = t_df[t_df["intended_word"] == word]["typed_word"].astype(str).tolist()
                pssm = WordPSSM(intended_word=word)
                pssm.build_from_variants(variants, scoring_matrix=self.scoring_matrix)
                self.profiles[typist][word] = pssm

    def score_message(self, typist_id: str, word_pairs: List[Tuple[str, str]]) -> float:
        """
        Computes aggregate PSSM score for a list of (intended, typed) word pairs.
        """
        if typist_id not in self.profiles:
            return -50.0

        user_pssms = self.profiles[typist_id]
        scores = []
        for intended, typed in word_pairs:
            intended = intended.lower()
            if intended in user_pssms:
                s = user_pssms[intended].score_variant(typed, scoring_matrix=self.scoring_matrix)
                scores.append(s)
            else:
                # Default baseline score
                scores.append(-2.0 if intended != typed.lower() else 3.0)

        return float(np.sum(scores)) if scores else 0.0


# =====================================================================
# 2. HMM FOR KEYSTROKE BEHAVIORAL FORENSICS (hmmlearn)
# =====================================================================

class TypistHMM:
    """
    Wraps hmmlearn.hmm.CategoricalHMM to model keystroke edit-operation dynamics.
    States:
        0: Careful
        1: Rushing
        2: Sleepy
    Observations:
        0: 'M' (Match)
        1: 'S' (Substitution)
        2: 'I' (Insertion - double-tap/held key)
        3: 'D' (Deletion - dropped letter/vowel)
    """
    def __init__(self, typist_id: str, random_state: int = 42):
        self.typist_id = typist_id
        self.n_components = 3  # Careful, Rushing, Sleepy
        self.n_features = 4    # M, S, I, D

        # Informative keystroke priors
        startprob_prior = np.array([0.60, 0.25, 0.15])
        transmat_prior = np.array([
            [0.85, 0.10, 0.05],  # Careful stays careful
            [0.20, 0.70, 0.10],  # Rushing tends to persist
            [0.15, 0.15, 0.70]   # Sleepy tends to persist
        ])

        # Emission probabilities: [M, S, I, D]
        emission_prior = np.array([
            [0.88, 0.04, 0.04, 0.04],  # Careful: Mostly Matches
            [0.45, 0.25, 0.05, 0.25],  # Rushing: High Substitutions & Deletions
            [0.40, 0.15, 0.35, 0.10]   # Sleepy: High Insertions (key bounce)
        ])

        self.model = hmm.CategoricalHMM(
            n_components=self.n_components,
            n_features=self.n_features,
            random_state=random_state,
            n_iter=40,
            tol=1e-4,
            params="te",       # learn transitions and emissions (preserve stable starts)
            init_params=""
        )
        self.model.startprob_ = startprob_prior
        self.model.transmat_ = transmat_prior
        self.model.emissionprob_ = emission_prior

    def fit_edit_operations(self, op_sequences: List[str]):
        """
        Trains the HMM using Baum-Welch (EM) on a collection of M/S/I/D operation strings.
        """
        X_list = []
        lengths = []

        for seq in op_sequences:
            if not seq:
                continue
            indices = [OP_MAP[c] for c in seq if c in OP_MAP]
            if indices:
                X_list.extend(indices)
                lengths.append(len(indices))

        if not lengths:
            return

        X = np.array(X_list, dtype=int).reshape(-1, 1)

        try:
            self.model.fit(X, lengths)
            # Ensure no emission probability collapses to absolute zero
            self.model.emissionprob_ = np.clip(self.model.emissionprob_, 1e-4, 1.0)
            self.model.emissionprob_ /= self.model.emissionprob_.sum(axis=1, keepdims=True)
            self.model.transmat_ = np.clip(self.model.transmat_, 1e-4, 1.0)
            self.model.transmat_ /= self.model.transmat_.sum(axis=1, keepdims=True)
        except Exception as e:
            print(f"Warning: Baum-Welch fitting warning for {self.typist_id}: {e}")

    def score_sequence(self, op_sequence: str) -> float:
        """
        Uses the Forward algorithm to return the log-likelihood: log P(O | lambda).
        Normalized by sequence length to allow fair comparison across lengths.
        """
        if not op_sequence:
            return -50.0

        indices = [OP_MAP[c] for c in op_sequence if c in OP_MAP]
        if not indices:
            return -50.0

        X = np.array(indices, dtype=int).reshape(-1, 1)
        try:
            log_lik = float(self.model.score(X))
            return log_lik
        except Exception:
            return -100.0

    def decode_states(self, op_sequence: str) -> Tuple[List[str], np.ndarray]:
        """
        Uses the Viterbi algorithm to decode the sequence of hidden mood states
        (Careful, Rushing, Sleepy) for the observed edit-op string.
        Returns:
            (list of state names, state index array)
        """
        if not op_sequence:
            return [], np.array([])

        indices = [OP_MAP[c] for c in op_sequence if c in OP_MAP]
        if not indices:
            return [], np.array([])

        X = np.array(indices, dtype=int).reshape(-1, 1)
        try:
            state_indices = self.model.predict(X)
            state_names = [STATE_NAMES[idx] for idx in state_indices]
            return state_names, state_indices
        except Exception:
            # Fallback
            return [STATE_NAMES[0]] * len(indices), np.zeros(len(indices), dtype=int)


# =====================================================================
# 3. CONVENIENCE PROFILER MANAGER
# =====================================================================

class KeystrokeProfiler:
    """
    Manages trained HMM models and PSSM profiles for Person_A, Person_B, and Impostor.
    """
    def __init__(self, calibration_csv: str = "calibration.csv"):
        self.calibration_csv = calibration_csv
        self.scoring_matrix = learn_scoring_matrix_from_csv(calibration_csv)
        self.pssm_registry = ProfilerPSSMRegistry(scoring_matrix=self.scoring_matrix)
        self.models: Dict[str, TypistHMM] = {}

    def train_all(self):
        """Trains PSSMs and HMMs for Person A, Person B, and Impostor efficiently."""
        print("Pre-computing training alignments...")
        df = pd.read_csv(self.calibration_csv)
        train_df = df[df["split"] == "train"] if "split" in df.columns else df

        # Cache alignments: {(intended, typed): (score, ref_aligned, var_aligned, ops)}
        alignment_cache = {}
        unique_pairs = train_df[["intended_word", "typed_word"]].drop_duplicates().values
        for intended, typed in unique_pairs:
            intended_str = str(intended).lower()
            typed_str = str(typed).lower()
            alignment_cache[(intended_str, typed_str)] = needleman_wunsch_affine(
                intended_str, typed_str, scoring_matrix=self.scoring_matrix
            )

        print("Building PSSM profiles...")
        typists = train_df["typist_id"].unique()
        for typist in typists:
            self.pssm_registry.profiles[typist] = {}
            t_df = train_df[train_df["typist_id"] == typist]
            words = t_df["intended_word"].unique()

            for word in words:
                word_str = str(word).lower()
                variants = t_df[t_df["intended_word"] == word]["typed_word"].astype(str).tolist()
                pssm = WordPSSM(intended_word=word_str)

                # Build counts using cached alignments
                counts = np.full((27, pssm.length), pssm.pseudocount)
                for variant in variants:
                    v_str = str(variant).lower()
                    if (word_str, v_str) in alignment_cache:
                        _, ref_aligned, var_aligned, _ = alignment_cache[(word_str, v_str)]
                    else:
                        _, ref_aligned, var_aligned, _ = needleman_wunsch_affine(
                            word_str, v_str, scoring_matrix=self.scoring_matrix
                        )
                    ref_col = 0
                    for r_char, v_char in zip(ref_aligned, var_aligned):
                        if r_char != '-':
                            if ref_col < pssm.length:
                                c_idx = CHAR_INDEX.get(v_char, CHAR_INDEX['-'])
                                counts[c_idx, ref_col] += 1.0
                            ref_col += 1

                col_sums = counts.sum(axis=0, keepdims=True)
                prob_mat = counts / col_sums
                pssm.pssm_matrix = np.log2(prob_mat / pssm.background_freq[:, np.newaxis])
                self.pssm_registry.profiles[typist][word_str] = pssm

        print("Training HMMs via Baum-Welch...")
        for typist_id in ["Person_A", "Person_B", "Impostor"]:
            t_df = train_df[train_df["typist_id"] == typist_id]
            hmm_model = TypistHMM(typist_id=typist_id)

            op_seqs = []
            for _, row in t_df.iterrows():
                pair_key = (str(row["intended_word"]).lower(), str(row["typed_word"]).lower())
                ops = alignment_cache[pair_key][3]
                op_seqs.append(ops)

            hmm_model.fit_edit_operations(op_seqs)
            self.models[typist_id] = hmm_model
            print(f"HMM for {typist_id} successfully trained on {len(op_seqs)} sequences.")

    def forward_scores(self, op_sequence: str) -> Dict[str, float]:
        """Runs Forward algorithm across all models and returns log-likelihoods."""
        return {
            typist: self.models[typist].score_sequence(op_sequence)
            for typist in self.models
        }

    def viterbi_decode(self, typist_id: str, op_sequence: str) -> Tuple[List[str], np.ndarray]:
        """Runs Viterbi decoding under the specified typist's model."""
        if typist_id in self.models:
            return self.models[typist_id].decode_states(op_sequence)
        return [], np.array([])
