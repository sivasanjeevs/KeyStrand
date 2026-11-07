"""
Keystrand: Keystroke Genomics & Forensic Hijack Detection
Module 2: The Forensic Aligner (aligner.py)

Algorithms implemented from scratch:
1. Global Needleman-Wunsch with Gotoh Affine Gap Penalties.
2. Local Smith-Waterman with Gotoh Affine Gap Penalties.
3. Scoring Matrix learned from calibration keystrokes (log-odds) + QWERTY distance matrix.
4. Statistical Significance Tester: Sequence shuffling + Gumbel distribution fitting (scipy).
5. Edit-operation extractor: converts alignments to M/S/I/D strings.
6. Dot Matrix generator for visual inspection of keystroke patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional


# =====================================================================
# 1. SCORING MATRICES: QWERTY & EMPIRICAL LOG-ODDS
# =====================================================================

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

# Keyboard layout for Euclidean coordinate distance
KEYBOARD_COORDS = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    'a': (1, 0.3), 's': (1, 1.3), 'd': (1, 2.3), 'f': (1, 3.3), 'g': (1, 4.3),
    'h': (1, 5.3), 'j': (1, 6.3), 'k': (1, 7.3), 'l': (1, 8.3),
    'z': (2, 0.7), 'x': (2, 1.7), 'c': (2, 2.7), 'v': (2, 3.7), 'b': (2, 4.7),
    'n': (2, 5.7), 'm': (2, 6.7)
}


def build_qwerty_scoring_matrix(match_score: float = 3.0, max_penalty: float = -3.0) -> Dict[Tuple[str, str], float]:
    """
    Builds a baseline substitution matrix based on physical distance on a QWERTY keyboard.
    Closer keys have lesser penalty; exact matches have match_score.
    """
    matrix = {}
    for a in ALPHABET:
        for b in ALPHABET:
            if a == b:
                matrix[(a, b)] = float(match_score)
            else:
                if a in KEYBOARD_COORDS and b in KEYBOARD_COORDS:
                    r1, c1 = KEYBOARD_COORDS[a]
                    r2, c2 = KEYBOARD_COORDS[b]
                    dist = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
                    # Scale distance: distance 1 => -0.5, distance 4+ => max_penalty
                    score = match_score - 1.5 * dist
                    matrix[(a, b)] = float(max(score, max_penalty))
                else:
                    matrix[(a, b)] = float(max_penalty)
    return matrix


def learn_scoring_matrix_from_csv(
    csv_path: str = "calibration.csv",
    pseudocount: float = 0.5,
    match_boost: float = 3.0
) -> Dict[Tuple[str, str], float]:
    """
    Learns an empirical log-odds substitution scoring matrix from observed typos in calibration.csv.
    S(a, b) = log2( q_ab / (p_a * p_b) )
    where q_ab is joint probability of intended a and typed b, and p is background frequency.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Fallback to pure QWERTY if file not yet present
        return build_qwerty_scoring_matrix()

    # Train split only to avoid data snooping
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
    else:
        train_df = df

    # Pairwise substitution counts
    counts = np.full((26, 26), pseudocount)

    # Initial naive pass or character-level alignment
    for _, row in train_df.iterrows():
        intended = str(row["intended_word"]).lower()
        typed = str(row["typed_word"]).lower()

        # Coarse alignment by position
        min_len = min(len(intended), len(typed))
        for k in range(min_len):
            c1, c2 = intended[k], typed[k]
            if c1 in CHAR_TO_IDX and c2 in CHAR_TO_IDX:
                counts[CHAR_TO_IDX[c1], CHAR_TO_IDX[c2]] += 1.0

    # Symmetrize or maintain directional joint distribution:
    # P(intended = a, typed = b)
    total_pairs = counts.sum()
    q_ab = counts / total_pairs

    # Background marginal frequencies
    p_intended = q_ab.sum(axis=1, keepdims=True)
    p_typed = q_ab.sum(axis=0, keepdims=True)

    expected = p_intended @ p_typed
    expected[expected == 0] = 1e-9

    log_odds = np.log2(q_ab / expected)

    # Scale and format as dictionary
    matrix = {}
    for a in ALPHABET:
        i = CHAR_TO_IDX[a]
        for b in ALPHABET:
            j = CHAR_TO_IDX[b]
            score = log_odds[i, j]
            # Ensure exact matches have solid positive reward
            if a == b and score < 1.0:
                score += match_boost
            matrix[(a, b)] = float(score)

    return matrix


# =====================================================================
# 2. PAIRWISE ALIGNMENT WITH GOTOH AFFINE GAP PENALTIES
# =====================================================================

NEG_INF = -1e9


def needleman_wunsch_affine(
    seq1: str,
    seq2: str,
    scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    gap_open: float = 2.5,
    gap_extend: float = 0.5
) -> Tuple[float, str, str, str]:
    """
    Global Needleman-Wunsch alignment with Gotoh Affine Gap Penalties.
    Cost of a gap of length k: gap_open + (k * gap_extend).
    Returns:
        (score, aligned_seq1, aligned_seq2, edit_operations)
        where edit_operations is a string of 'M', 'S', 'I', 'D'.
        - 'M': Match
        - 'S': Substitution
        - 'D': Deletion (character present in seq1/intended, absent in seq2/typed)
        - 'I': Insertion (character extra in seq2/typed)
    """
    if scoring_matrix is None:
        scoring_matrix = build_qwerty_scoring_matrix()

    n = len(seq1)
    m = len(seq2)

    # Three Gotoh DP matrices
    # M_mat[i, j]: alignment ending with seq1[i-1] aligned to seq2[j-1]
    # Ix[i, j]: alignment ending with deletion in seq1 (gap in seq2)
    # Iy[i, j]: alignment ending with insertion in seq2 (gap in seq1)
    M_mat = np.full((n + 1, m + 1), NEG_INF)
    Ix = np.full((n + 1, m + 1), NEG_INF)
    Iy = np.full((n + 1, m + 1), NEG_INF)

    # Traceback matrices:
    # 0: M, 1: Ix, 2: Iy
    trace_M = np.zeros((n + 1, m + 1), dtype=int)
    trace_Ix = np.zeros((n + 1, m + 1), dtype=int)
    trace_Iy = np.zeros((n + 1, m + 1), dtype=int)

    # Base cases
    M_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        Ix[i, 0] = -(gap_open + i * gap_extend)
        M_mat[i, 0] = Ix[i, 0]
        trace_Ix[i, 0] = 1

    for j in range(1, m + 1):
        Iy[0, j] = -(gap_open + j * gap_extend)
        M_mat[0, j] = Iy[0, j]
        trace_Iy[0, j] = 2

    for i in range(1, n + 1):
        c1 = seq1[i - 1].lower()
        for j in range(1, m + 1):
            c2 = seq2[j - 1].lower()
            sub_score = scoring_matrix.get((c1, c2), -2.0)

            # Update Ix (gap in seq2 / deletion of seq1)
            cost_from_m = M_mat[i - 1, j] - (gap_open + gap_extend)
            cost_from_ix = Ix[i - 1, j] - gap_extend
            if cost_from_m >= cost_from_ix:
                Ix[i, j] = cost_from_m
                trace_Ix[i, j] = 0
            else:
                Ix[i, j] = cost_from_ix
                trace_Ix[i, j] = 1

            # Update Iy (gap in seq1 / insertion in seq2)
            cost_from_m = M_mat[i, j - 1] - (gap_open + gap_extend)
            cost_from_iy = Iy[i, j - 1] - gap_extend
            if cost_from_m >= cost_from_iy:
                Iy[i, j] = cost_from_m
                trace_Iy[i, j] = 0
            else:
                Iy[i, j] = cost_from_iy
                trace_Iy[i, j] = 2

            # Update M_mat (match or substitution)
            prev_m = M_mat[i - 1, j - 1]
            prev_ix = Ix[i - 1, j - 1]
            prev_iy = Iy[i - 1, j - 1]

            best_prev = max(prev_m, prev_ix, prev_iy)
            M_mat[i, j] = sub_score + best_prev

            if best_prev == prev_m:
                trace_M[i, j] = 0
            elif best_prev == prev_ix:
                trace_M[i, j] = 1
            else:
                trace_M[i, j] = 2

    # Termination: choose best end state among M, Ix, Iy
    final_scores = [M_mat[n, m], Ix[n, m], Iy[n, m]]
    best_state = int(np.argmax(final_scores))
    final_score = float(final_scores[best_state])

    # Traceback
    align1 = []
    align2 = []
    ops = []

    i, j = n, m
    curr_state = best_state

    while i > 0 or j > 0:
        if i == 0:
            align1.append('-')
            align2.append(seq2[j - 1])
            ops.append('I')
            j -= 1
        elif j == 0:
            align1.append(seq1[i - 1])
            align2.append('-')
            ops.append('D')
            i -= 1
        elif curr_state == 0:  # M state (match / substitution)
            prev_state = trace_M[i, j]
            c1 = seq1[i - 1]
            c2 = seq2[j - 1]
            align1.append(c1)
            align2.append(c2)
            ops.append('M' if c1.lower() == c2.lower() else 'S')
            i -= 1
            j -= 1
            curr_state = prev_state
        elif curr_state == 1:  # Ix state (gap in seq2 / deletion of seq1)
            prev_state = trace_Ix[i, j]
            align1.append(seq1[i - 1])
            align2.append('-')
            ops.append('D')
            i -= 1
            curr_state = prev_state
        else:  # Iy state (gap in seq1 / insertion in seq2)
            prev_state = trace_Iy[i, j]
            align1.append('-')
            align2.append(seq2[j - 1])
            ops.append('I')
            j -= 1
            curr_state = prev_state

    align1.reverse()
    align2.reverse()
    ops.reverse()

    return final_score, "".join(align1), "".join(align2), "".join(ops)


def smith_waterman_affine(
    seq1: str,
    seq2: str,
    scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    gap_open: float = 2.5,
    gap_extend: float = 0.5
) -> Tuple[float, str, str, str, Tuple[int, int], Tuple[int, int]]:
    """
    Local Smith-Waterman alignment with Gotoh Affine Gap Penalties.
    Finds optimal local subword alignment (e.g. finds 'tomorrow' inside 'xxtomorrowqq').
    Returns:
        (max_score, aligned_seq1, aligned_seq2, ops, (start_i, end_i), (start_j, end_j))
    """
    if scoring_matrix is None:
        scoring_matrix = build_qwerty_scoring_matrix()

    n = len(seq1)
    m = len(seq2)

    M_mat = np.zeros((n + 1, m + 1))
    Ix = np.full((n + 1, m + 1), NEG_INF)
    Iy = np.full((n + 1, m + 1), NEG_INF)

    trace_M = np.zeros((n + 1, m + 1), dtype=int)
    trace_Ix = np.zeros((n + 1, m + 1), dtype=int)
    trace_Iy = np.zeros((n + 1, m + 1), dtype=int)

    max_score = 0.0
    max_pos = (0, 0)
    best_state = 0

    for i in range(1, n + 1):
        c1 = seq1[i - 1].lower()
        for j in range(1, m + 1):
            c2 = seq2[j - 1].lower()
            sub_score = scoring_matrix.get((c1, c2), -2.0)

            # Ix
            cost_from_m = M_mat[i - 1, j] - (gap_open + gap_extend)
            cost_from_ix = Ix[i - 1, j] - gap_extend
            if cost_from_m >= cost_from_ix:
                Ix[i, j] = cost_from_m
                trace_Ix[i, j] = 0
            else:
                Ix[i, j] = cost_from_ix
                trace_Ix[i, j] = 1

            # Iy
            cost_from_m = M_mat[i, j - 1] - (gap_open + gap_extend)
            cost_from_iy = Iy[i, j - 1] - gap_extend
            if cost_from_m >= cost_from_iy:
                Iy[i, j] = cost_from_m
                trace_Iy[i, j] = 0
            else:
                Iy[i, j] = cost_from_iy
                trace_Iy[i, j] = 2

            # M
            prev_m = M_mat[i - 1, j - 1]
            prev_ix = Ix[i - 1, j - 1]
            prev_iy = Iy[i - 1, j - 1]
            best_prev = max(prev_m, prev_ix, prev_iy)

            val = sub_score + best_prev
            # Local threshold at 0
            if val <= 0:
                M_mat[i, j] = 0.0
                trace_M[i, j] = -1
            else:
                M_mat[i, j] = val
                if best_prev == prev_m:
                    trace_M[i, j] = 0
                elif best_prev == prev_ix:
                    trace_M[i, j] = 1
                else:
                    trace_M[i, j] = 2

            if M_mat[i, j] > max_score:
                max_score = float(M_mat[i, j])
                max_pos = (i, j)
                best_state = 0

    # Traceback from max_pos until score reaches 0
    i, j = max_pos
    end_i, end_j = i, j
    curr_state = best_state

    align1 = []
    align2 = []
    ops = []

    while i > 0 and j > 0 and M_mat[i, j] > 0:
        if curr_state == 0:
            prev_state = trace_M[i, j]
            if prev_state == -1:
                break
            c1 = seq1[i - 1]
            c2 = seq2[j - 1]
            align1.append(c1)
            align2.append(c2)
            ops.append('M' if c1.lower() == c2.lower() else 'S')
            i -= 1
            j -= 1
            curr_state = prev_state
        elif curr_state == 1:
            prev_state = trace_Ix[i, j]
            align1.append(seq1[i - 1])
            align2.append('-')
            ops.append('D')
            i -= 1
            curr_state = prev_state
        else:
            prev_state = trace_Iy[i, j]
            align1.append('-')
            align2.append(seq2[j - 1])
            ops.append('I')
            j -= 1
            curr_state = prev_state

    align1.reverse()
    align2.reverse()
    ops.reverse()

    start_i, start_j = i, j
    return max_score, "".join(align1), "".join(align2), "".join(ops), (start_i, end_i), (start_j, end_j)


# =====================================================================
# 3. STATISTICAL SIGNIFICANCE & GUMBEL DISTRIBUTION FITTING
# =====================================================================

def compute_alignment_significance(
    seq1: str,
    seq2: str,
    scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    n_shuffles: int = 100,
    gap_open: float = 2.5,
    gap_extend: float = 0.5
) -> Dict[str, float]:
    """
    Shuffles seq2 n_shuffles times, aligns each permuted sequence against seq1,
    fits the null score distribution to a Gumbel (extreme value) distribution,
    and returns both the empirical and fitted p-values.
    """
    real_score, _, _, _ = needleman_wunsch_affine(
        seq1, seq2, scoring_matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend
    )

    seq2_chars = list(seq2)
    null_scores = []

    for _ in range(n_shuffles):
        shuffled = "".join(np.random.permutation(seq2_chars))
        s, _, _, _ = needleman_wunsch_affine(
            seq1, shuffled, scoring_matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend
        )
        null_scores.append(s)

    null_scores = np.array(null_scores)
    empirical_p = float(np.sum(null_scores >= real_score) + 1.0) / (n_shuffles + 1.0)

    # Fit Gumbel (Right-skewed Extreme Value Distribution Type I)
    # CDF: F(x) = exp(-exp(-(x - loc) / scale))
    # p-value = 1 - F(real_score)
    try:
        loc, scale = stats.gumbel_r.fit(null_scores)
        fitted_p = float(stats.gumbel_r.sf(real_score, loc=loc, scale=scale))
    except Exception:
        loc, scale = float(np.mean(null_scores)), float(np.std(null_scores) + 1e-5)
        # Normal approximation fallback
        fitted_p = float(1.0 - stats.norm.cdf(real_score, loc=loc, scale=scale))

    return {
        "real_score": float(real_score),
        "mean_null": float(np.mean(null_scores)),
        "std_null": float(np.std(null_scores)),
        "gumbel_loc": float(loc),
        "gumbel_scale": float(scale),
        "empirical_p_value": float(empirical_p),
        "fitted_p_value": float(max(1e-6, min(1.0, fitted_p)))
    }


# =====================================================================
# 4. EDIT OPERATION STRING GENERATOR & DOT MATRIX VISUALIZER
# =====================================================================

def get_edit_operations(
    intended_word: str,
    typed_word: str,
    scoring_matrix: Optional[Dict[Tuple[str, str], float]] = None
) -> str:
    """
    Takes an intended word and a typed word, executes Needleman-Wunsch with
    affine gaps, and outputs the edit-operation string consisting of 'M', 'S', 'I', 'D'.
    """
    _, _, _, ops = needleman_wunsch_affine(intended_word, typed_word, scoring_matrix=scoring_matrix)
    return ops


def generate_dot_matrix(seq1: str, seq2: str) -> np.ndarray:
    """
    Generates a 2D binary matrix where matrix[i, j] = 1 if seq1[i] == seq2[j] else 0.
    Used for dot-plot visualizations of sequence similarity.
    """
    mat = np.zeros((len(seq1), len(seq2)), dtype=int)
    for i, c1 in enumerate(seq1):
        for j, c2 in enumerate(seq2):
            if c1.lower() == c2.lower():
                mat[i, j] = 1
    return mat
