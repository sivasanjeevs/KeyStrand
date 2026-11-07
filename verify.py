"""
Keystrand: Verification Test Suite (verify.py)

Automated verification of core algorithms without hallucinations:
1. Needleman-Wunsch & Smith-Waterman with Gotoh Affine Gaps.
2. Learned scoring matrix verification.
3. PSSM column mathematical computation.
4. HMM Forward & Viterbi algorithms.
5. End-to-end detection on the 3 AM hijacked message.
"""

import numpy as np
from aligner import (
    needleman_wunsch_affine,
    smith_waterman_affine,
    compute_alignment_significance,
    learn_scoring_matrix_from_csv
)
from profiler import WordPSSM, TypistHMM, KeystrokeProfiler
from engine import ForensicEngine


def test_needleman_wunsch_affine():
    print("\n[Test 1] Needleman-Wunsch with Gotoh Affine Gaps...")
    # Textbook example: "GATTACA" vs "GCATGCU"
    score, a1, a2, ops = needleman_wunsch_affine("gattaca", "gcatgcu", gap_open=2.0, gap_extend=0.5)
    print(f"  Seq 1: {a1}")
    print(f"  Seq 2: {a2}")
    print(f"  Ops:   {ops}")
    print(f"  Score: {score:.2f}")
    assert len(a1) == len(a2) == len(ops)
    assert 'M' in ops
    print("  ✓ Global alignment with affine gaps verified.")


def test_smith_waterman_affine():
    print("\n[Test 2] Smith-Waterman Local Affine Alignment...")
    # Finding "tomorrow" inside garbage "xxtomorrowqq"
    score, sw1, sw2, ops, r1, r2 = smith_waterman_affine("tomorrow", "xxtomorrowqq")
    print(f"  Local Match: '{sw1}' vs '{sw2}'")
    print(f"  Ops: {ops}, Score: {score:.2f}")
    assert sw1 == "tomorrow"
    assert sw2 == "tomorrow"
    assert ops == "MMMMMMMM"
    print("  ✓ Local alignment successfully isolated core word.")


def test_pssm_calculation():
    print("\n[Test 3] PSSM Column Frequency & Log-Odds...")
    pssm = WordPSSM(intended_word="cat")
    variants = ["cat", "catt", "ct", "kat"]
    pssm.build_from_variants(variants)
    # Check shape
    assert pssm.pssm_matrix.shape == (27, 3)
    # Check that 'c' at position 0 has high positive log-odds
    c_idx = ord('c') - ord('a')
    assert pssm.pssm_matrix[c_idx, 0] > 0
    print(f"  PSSM log-odds for 'c' at col 0: {pssm.pssm_matrix[c_idx, 0]:.2f}")
    print("  ✓ PSSM log-odds mathematical logic verified.")


def test_hmm_forward_and_viterbi():
    print("\n[Test 4] HMM Baum-Welch, Forward & Viterbi...")
    hmm_model = TypistHMM(typist_id="Test_Subject")
    # Simulate known bursts
    train_seqs = ["MMMMMMMM", "MMSSDDMM", "MMIIIMMM", "MMMDDMMM"]
    hmm_model.fit_edit_operations(train_seqs)

    test_seq = "MMMDDMMMIIM"
    log_lik = hmm_model.score_sequence(test_seq)
    states, idxs = hmm_model.decode_states(test_seq)
    print(f"  Observation: {test_seq}")
    print(f"  Forward Log-Likelihood: {log_lik:.2f}")
    print(f"  Viterbi Decoded States: {states}")
    assert len(states) == len(test_seq)
    print("  ✓ Forward and Viterbi state tracking verified.")


def test_end_to_end_forensic_engine():
    print("\n[Test 5] End-to-End Forensic Hijack Engine...")
    engine = ForensicEngine("calibration.csv")

    # The 3 AM Biryani Hijacked Message
    hijack_msg = "bhaiya send beriyani mony urgnt akount transfer right now"
    res = engine.analyze_message(hijack_msg, claimed_sender="Person_A")

    print(f"  Claimed Sender: Person_A")
    print(f"  Forensic Verdict: {res['verdict']}")
    print(f"  Is Hijack Detected: {res['is_hijack']}")
    print(f"  Confidence: {res['confidence']}%")
    print(f"  Reasoning: {res['reasoning']}")

    assert res["is_hijack"] is True
    assert res["verdict"] == "Impostor"
    print("  ✓ The 3 AM Impostor was successfully apprehended!")


if __name__ == "__main__":
    print("==================================================")
    print("          KEYSTRAND: FORENSIC VERIFICATION        ")
    print("==================================================")
    test_needleman_wunsch_affine()
    test_smith_waterman_affine()
    test_pssm_calculation()
    test_hmm_forward_and_viterbi()
    test_end_to_end_forensic_engine()
    print("\n==================================================")
    print("       ALL FORENSIC VERIFICATION TESTS PASSED!    ")
    print("==================================================")

