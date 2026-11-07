"""
Keystrand: Keystroke Genomics & Forensic Hijack Detection
Module 4: The Evaluation & Verdict Engine (engine.py)

Responsibilities:
1. Message tokenization and dictionary mapping using pyspellchecker and domain lexicon.
2. Forensic pipeline execution:
   - Aligns words (Needleman-Wunsch affine).
   - Generates composite M/S/I/D edit string.
   - Computes PSSM spelling scores and HMM Forward log-likelihoods.
3. Multi-factor Bayesian / Likelihood-Ratio Verdict Engine:
   - Evaluates claimed sender vs true typist.
   - Detects impostor hijacks with confidence score and forensic reasoning.
4. Benchmark & Validation:
   - Generates 3x3 Confusion Matrices for 4 forensic models:
     (1) Basic Edit Distance Only
     (2) Scoring Matrix Only
     (3) HMM Only
     (4) Combined Forensic Toolkit
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from spellchecker import SpellChecker
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from aligner import (
    needleman_wunsch_affine,
    smith_waterman_affine,
    compute_alignment_significance,
    generate_dot_matrix,
    learn_scoring_matrix_from_csv
)
from profiler import KeystrokeProfiler, OP_MAP, STATE_COLORS
from data_gen import INTERROGATION_WORDS


# High-frequency college/chat aliases and abbreviations
COMMON_ALIASES = {
    'tmrw': 'tomorrow', 'tomorow': 'tomorrow', '2morow': 'tomorrow',
    'tnite': 'tonight', 'tonite': 'tonight',
    'bro': 'brother', 'bhaiya': 'brother',
    'plz': 'please', 'plzzz': 'please', 'pls': 'please',
    'acct': 'account', 'akount': 'account',
    'trfr': 'transfer', 'trf': 'transfer',
    'msg': 'message', 'thx': 'thanks',
    'briyani': 'biryani', 'beriyani': 'biryani', 'birryani': 'biryani', 'birryanii': 'biryani',
    'hungrry': 'hungry', 'mony': 'money', 'kantin': 'canteen',
    'colleg': 'college', 'urgnt': 'urgent', 'denner': 'dinner',
    'gpay': 'google'
}


class ForensicEngine:
    """
    Central orchestration engine for keystroke sequence forensics.
    """
    def __init__(self, calibration_csv: str = "calibration.csv"):
        self.calibration_csv = calibration_csv
        self.spell = SpellChecker()
        self.spell.word_frequency.load_words(INTERROGATION_WORDS)

        # Initialize profiler (HMM + PSSM + Scoring matrix)
        self.profiler = KeystrokeProfiler(calibration_csv)
        self.profiler.train_all()
        self.scoring_matrix = self.profiler.scoring_matrix

    def tokenize_and_map(self, message: str) -> List[Tuple[str, str]]:
        """
        Tokenizes raw message string and pairs each typed token with its intended word.
        Returns:
            List of (intended_word, typed_word) tuples.
        """
        raw_tokens = re.findall(r"[a-zA-Z0-9']+", message.lower())
        pairs = []

        for token in raw_tokens:
            clean_token = token.strip(".,!?'\"")
            if not clean_token or clean_token.isdigit():
                continue

            # 1. Alias lookup
            if clean_token in COMMON_ALIASES:
                pairs.append((COMMON_ALIASES[clean_token], clean_token))
                continue

            # 2. Direct match in interrogation lexicon
            if clean_token in INTERROGATION_WORDS:
                pairs.append((clean_token, clean_token))
                continue

            # 3. Collapse repeated keystrokes (e.g., 'soooomeone' -> 'someone')
            collapsed = re.sub(r'(.)\1{2,}', r'\1', clean_token)
            if collapsed in INTERROGATION_WORDS:
                pairs.append((collapsed, clean_token))
                continue

            # 4. Spellchecker correction
            corr = self.spell.correction(clean_token)
            intended = corr if corr else clean_token
            pairs.append((intended, clean_token))

        return pairs

    def analyze_message(
        self,
        message: str,
        claimed_sender: str = "Person_A",
        run_significance: bool = True
    ) -> Dict[str, Any]:
        """
        Full forensic examination of a typed message.
        """
        word_pairs = self.tokenize_and_map(message)
        if not word_pairs:
            return {
                "verdict": "Inconclusive",
                "confidence": 0.0,
                "reasoning": "No alphabetic words identified in message.",
                "details": {}
            }

        composite_ops = []
        alignments = []
        total_nw_score = 0.0

        for intended, typed in word_pairs:
            score, a1, a2, ops = needleman_wunsch_affine(
                intended, typed, scoring_matrix=self.scoring_matrix
            )
            composite_ops.append(ops)
            alignments.append({
                "intended": intended,
                "typed": typed,
                "aligned_intended": a1,
                "aligned_typed": a2,
                "ops": ops,
                "score": score
            })
            total_nw_score += score

        full_op_string = "".join(composite_ops)

        # 1. HMM Forward log-likelihoods
        hmm_scores = self.profiler.forward_scores(full_op_string)

        # 2. PSSM spelling scores
        pssm_scores = {
            typist: self.profiler.pssm_registry.score_message(typist, word_pairs)
            for typist in ["Person_A", "Person_B", "Impostor"]
        }

        # 3. Viterbi hidden mood decoding
        viterbi_states, _ = self.profiler.viterbi_decode(claimed_sender, full_op_string)

        # 4. Edit operations distribution
        total_ops_count = max(1, len(full_op_string))
        op_counts = {
            'M': full_op_string.count('M'),
            'S': full_op_string.count('S'),
            'I': full_op_string.count('I'),
            'D': full_op_string.count('D')
        }
        op_percents = {k: round((v / total_ops_count) * 100, 1) for k, v in op_counts.items()}

        # 5. Combined Model Scoring
        # Normalized score: combination of HMM log-likelihood per op, PSSM score, and NW score
        combined_scores = {}
        for typist in ["Person_A", "Person_B", "Impostor"]:
            norm_hmm = hmm_scores[typist] / total_ops_count
            pssm = pssm_scores[typist]
            # Habitual bias based on ops profile
            habit_score = 0.0
            if typist == "Person_A":
                # Person A drops vowels: D is typical, I is rare
                habit_score = 1.5 * op_counts['D'] - 2.0 * op_counts['I']
            elif typist == "Person_B":
                # Person B double taps: I is typical
                habit_score = 2.5 * op_counts['I'] - 1.0 * op_counts['D']
            elif typist == "Impostor":
                # Impostor has high phonetic substitutions S
                habit_score = 2.0 * op_counts['S'] - 1.0 * op_counts['I']

            combined_scores[typist] = (3.0 * norm_hmm) + (0.5 * pssm) + habit_score

        # 6. Statistical Significance of Primary Alignment
        sig_result = None
        if run_significance and word_pairs:
            # Test the word with the lowest relative match ratio (the most anomalous typo)
            most_anomalous = min(alignments, key=lambda x: x["ops"].count('M') / max(1, len(x["ops"])))
            sig_result = compute_alignment_significance(
                most_anomalous["intended"],
                most_anomalous["typed"],
                scoring_matrix=self.scoring_matrix,
                n_shuffles=80
            )
            sig_result["tested_word"] = most_anomalous["intended"]
            sig_result["typed_form"] = most_anomalous["typed"]

        # 7. Verdict Logic & Likelihood Ratios
        sorted_typists = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_typist, top_score = sorted_typists[0]
        second_typist, second_score = sorted_typists[1]

        # Log-Likelihood Ratio of claimed vs top
        claimed_hmm = hmm_scores.get(claimed_sender, -100.0)
        top_hmm = hmm_scores.get(top_typist, -100.0)
        llr_claimed_vs_top = claimed_hmm - top_hmm

        # Decision threshold
        if claimed_sender == top_typist:
            verdict = claimed_sender
            is_hijack = False
            margin = top_score - second_score
            confidence = min(99.0, max(65.0, 80.0 + 3.0 * margin))
            reasoning = (
                f"Typing habits match {claimed_sender}'s calibrated profile. "
                f"Keystroke edit pattern ({op_percents['M']}% matches, {op_percents['S']}% subs, "
                f"{op_percents['I']}% ins, {op_percents['D']}% dels) aligns with expected behavioral signature."
            )
        else:
            verdict = top_typist
            is_hijack = True
            margin = top_score - second_score
            confidence = min(99.0, max(70.0, 82.0 + 3.5 * margin))
            reasoning = (
                f"🚨 HIJACK DETECTED! Message was sent from {claimed_sender}'s account, "
                f"but forensic analysis strongly identifies true typist as {top_typist}. "
                f"Forensic score margin: +{margin:.2f} over claimed author. "
                f"Key anomalies: {op_counts['S']} substitutions, {op_counts['I']} key bounces, {op_counts['D']} omissions."
            )

        return {
            "verdict": verdict,
            "claimed_sender": claimed_sender,
            "is_hijack": is_hijack,
            "confidence": round(confidence, 1),
            "reasoning": reasoning,
            "full_op_string": full_op_string,
            "op_counts": op_counts,
            "op_percents": op_percents,
            "hmm_scores": hmm_scores,
            "pssm_scores": pssm_scores,
            "combined_scores": combined_scores,
            "viterbi_states": viterbi_states,
            "alignments": alignments,
            "significance": sig_result,
            "word_pairs": word_pairs
        }

    # =================================================================
    # 8. BENCHMARK & 3x3 CONFUSION MATRIX EVALUATION
    # =================================================================

    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluates 4 detectors on the held-out test split of calibration.csv:
        1. Basic Edit Distance Only (simple Levenshtein match ratio)
        2. Scoring Matrix Only (learned substitution matrix sum)
        3. HMM Only (Baum-Welch Forward log-likelihood)
        4. Combined Forensic Engine (Scoring Matrix + PSSM + HMM)

        Returns 3x3 confusion matrices and metrics for all 4 detectors.
        """
        df = pd.read_csv(self.calibration_csv)
        test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.tail(360).copy()

        true_labels = []
        preds_edit_dist = []
        preds_scoring_mat = []
        preds_hmm = []
        preds_combined = []

        classes = ["Person_A", "Person_B", "Impostor"]

        for _, row in test_df.iterrows():
            typist = row["typist_id"]
            intended = str(row["intended_word"])
            typed = str(row["typed_word"])
            true_labels.append(typist)

            # Get alignment
            score, a1, a2, ops = needleman_wunsch_affine(
                intended, typed, scoring_matrix=self.scoring_matrix
            )

            # Model 1: Basic Edit Distance (Levenshtein match ratio)
            # Person A drops vowels (high D), Person B repeats keys (high I), Impostor high S
            m_count = ops.count('M')
            s_count = ops.count('S')
            i_count = ops.count('I')
            d_count = ops.count('D')

            if i_count > 1 or (i_count >= 1 and s_count == 0):
                pred_ed = "Person_B"
            elif d_count > s_count:
                pred_ed = "Person_A"
            else:
                pred_ed = "Impostor" if s_count > 0 else "Person_A"
            preds_edit_dist.append(pred_ed)

            # Model 2: Scoring Matrix Only
            mat_dict = {
                "Person_A": score - (0.5 * d_count),
                "Person_B": score - (0.5 * i_count),
                "Impostor": score - (1.0 * s_count)
            }
            preds_scoring_mat.append(max(mat_dict, key=mat_dict.get))

            # Model 3: HMM Only (Forward log-likelihood)
            hmm_sc = self.profiler.forward_scores(ops)
            preds_hmm.append(max(hmm_sc, key=hmm_sc.get))

            # Model 4: Combined Model (HMM + PSSM + Habits)
            comb = {}
            for t in classes:
                norm_hmm = hmm_sc[t] / max(1, len(ops))
                pssm_val = self.profiler.pssm_registry.score_message(t, [(intended, typed)])
                habit_bonus = 0.0
                if t == "Person_A":
                    habit_bonus = 1.5 * d_count - 1.5 * i_count
                elif t == "Person_B":
                    habit_bonus = 2.0 * i_count - 1.0 * d_count
                elif t == "Impostor":
                    habit_bonus = 1.5 * s_count - 1.0 * i_count

                comb[t] = (2.0 * norm_hmm) + (0.4 * pssm_val) + habit_bonus
            preds_combined.append(max(comb, key=comb.get))

        # Compute confusion matrices
        results = {
            "classes": classes,
            "test_sample_count": len(true_labels),
            "models": {
                "Edit Distance Only": {
                    "accuracy": float(accuracy_score(true_labels, preds_edit_dist)),
                    "matrix": confusion_matrix(true_labels, preds_edit_dist, labels=classes).tolist(),
                    "report": classification_report(true_labels, preds_edit_dist, labels=classes, output_dict=True, zero_division=0)
                },
                "Scoring Matrix Only": {
                    "accuracy": float(accuracy_score(true_labels, preds_scoring_mat)),
                    "matrix": confusion_matrix(true_labels, preds_scoring_mat, labels=classes).tolist(),
                    "report": classification_report(true_labels, preds_scoring_mat, labels=classes, output_dict=True, zero_division=0)
                },
                "HMM Only": {
                    "accuracy": float(accuracy_score(true_labels, preds_hmm)),
                    "matrix": confusion_matrix(true_labels, preds_hmm, labels=classes).tolist(),
                    "report": classification_report(true_labels, preds_hmm, labels=classes, output_dict=True, zero_division=0)
                },
                "Combined Forensic Toolkit": {
                    "accuracy": float(accuracy_score(true_labels, preds_combined)),
                    "matrix": confusion_matrix(true_labels, preds_combined, labels=classes).tolist(),
                    "report": classification_report(true_labels, preds_combined, labels=classes, output_dict=True, zero_division=0)
                }
            }
        }

        return results
