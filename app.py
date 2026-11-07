"""
Keystrand: Keystroke Genomics & Forensic Hijack Detection
Module 5: The Streamlit Dashboard (app.py)

A cyber-forensics web dashboard to detect keystroke anomalies,
align sequences with affine gap penalties, decode mood with Viterbi,
and pinpoint impostors hijacking user chat accounts.
"""

import json
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from data_gen import build_and_save_data
from aligner import needleman_wunsch_affine, smith_waterman_affine, generate_dot_matrix
from profiler import STATE_COLORS
from engine import ForensicEngine

# Set Streamlit page configuration
st.set_page_config(
    page_title="Keystrand — Keystroke Forensics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for forensics dashboard
st.markdown("""
<style>
    .main-title {
        font-family: 'Courier New', monospace;
        font-weight: 800;
        color: #e74c3c;
        margin-bottom: 0px;
    }
    .sub-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .verdict-box-danger {
        background: linear-gradient(135deg, #4a1010 0%, #1e0505 100%);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
        margin-bottom: 20px;
    }
    .verdict-box-success {
        background: linear-gradient(135deg, #0d3b1e 0%, #061e0e 100%);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4);
        margin-bottom: 20px;
    }
    .chat-bubble-claimed {
        border-left: 4px solid #3498db;
        background-color: rgba(52, 152, 219, 0.1);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .chat-bubble-hijack {
        border-left: 4px solid #e74c3c;
        background-color: rgba(231, 76, 60, 0.15);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    .viterbi-chip {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 0.95rem;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# Cache Forensic Engine instantiation to avoid retraining on every interaction
@st.cache_resource(show_spinner="Initializing Forensic Engine (Training HMMs & PSSM Registry)...")
def get_forensic_engine(csv_path: str = "calibration.csv") -> ForensicEngine:
    if not os.path.exists(csv_path):
        build_and_save_data(csv_path=csv_path)
    return ForensicEngine(calibration_csv=csv_path)


@st.cache_data(show_spinner="Evaluating 4 Forensic Detectors on Test Set...")
def get_model_evaluations(_engine: ForensicEngine) -> Dict[str, Any]:
    return _engine.evaluate_models()


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/detective.png", width=64)
st.sidebar.title("Forensic Ops")

# Ensure base data exists
if not os.path.exists("calibration.csv") or not os.path.exists("chat_history.json"):
    with st.sidebar.status("Generating calibration and chat datasets..."):
        build_and_save_data()

# Data selector
chat_file_option = st.sidebar.selectbox(
    "Chat Session Source",
    ["chat_history.json (Default Investigation)", "Upload Custom JSON"]
)

chat_data: List[Dict] = []
if chat_file_option == "Upload Custom JSON":
    uploaded = st.sidebar.file_uploader("Upload chat JSON file", type=["json"])
    if uploaded is not None:
        try:
            chat_data = json.load(uploaded)
            st.sidebar.success(f"Loaded {len(chat_data)} messages.")
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")
    else:
        st.sidebar.info("Upload a JSON or switch back to default.")
else:
    with open("chat_history.json", "r", encoding="utf-8") as f:
        chat_data = json.load(f)

st.sidebar.markdown("---")
st.sidebar.subheader("Forensic Pipeline Status")
st.sidebar.markdown("""
- **Gotoh Affine Gap NW & SW**: `Ready`
- **Typo Log-Odds Matrix**: `Active (calibration.csv)`
- **PSSM Profiler**: `3 subjects x 40 words`
- **HMM Models**: `3x Baum-Welch (40 iter)`
""")

if st.sidebar.button("Re-calibrate & Re-train Models"):
    st.cache_resource.clear()
    st.cache_data.clear()
    build_and_save_data()
    st.sidebar.success("Models recalibrated and reset!")
    st.rerun()

# Load Engine
engine = get_forensic_engine("calibration.csv")


# =====================================================================
# MAIN HEADER
# =====================================================================

st.markdown("<h1 class='main-title'>🧬 KEYSTRAND</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Keystroke Genomics & Forensic Hijack Detection Engine</div>", unsafe_allow_html=True)

# Tabs
tab_investigation, tab_benchmark, tab_sandbox = st.tabs([
    "🕵️ Chat Investigation",
    "📊 Detector Benchmarks & Confusion Matrices",
    "🧪 Interactive Forensic Sandbox"
])


# =====================================================================
# TAB 1: CHAT INVESTIGATION & MESSAGE FORENSICS
# =====================================================================

with tab_investigation:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("💬 Group Chat History")
        st.caption("Select a message to run deep bioinformatics keystroke analysis.")

        msg_options = [
            f"#{m['id']} [{m['timestamp'][11:16]}] {m['sender_claimed']}: {m['text'][:38]}..."
            for m in chat_data
        ]

        # Preset selection default to message 4 (the hijacked biryani ransom)
        default_index = 3 if len(chat_data) >= 4 else 0
        selected_label = st.radio("Select Target Message:", msg_options, index=default_index)
        selected_id = int(selected_label.split(" ")[0].replace("#", ""))

        target_message = next(m for m in chat_data if m["id"] == selected_id)

        # Render chat timeline
        for m in chat_data:
            is_selected = (m["id"] == selected_id)
            is_hijack_known = (m.get("actual_author") != m.get("sender_claimed"))
            bubble_class = "chat-bubble-hijack" if is_hijack_known else "chat-bubble-claimed"
            border_note = " 👈 [SELECTED]" if is_selected else ""

            author_tag = f"<b>{m['sender_claimed']}</b>"
            if is_hijack_known:
                author_tag += " <span style='color:#e74c3c; font-size:0.8rem;'>(Compromised Session)</span>"

            st.markdown(f"""
            <div class='{bubble_class}'>
                <div style='font-size:0.8rem; color:#888;'>#{m['id']} • {m['timestamp']} {border_note}</div>
                <div>{author_tag}</div>
                <div style='font-size:1.05rem; margin-top:4px;'>"{m['text']}"</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.subheader(f"🔬 Forensics View: Message #{target_message['id']}")

        # Run forensic analysis on target message
        with st.spinner("Analyzing keystroke edit sequence..."):
            forensic_res = engine.analyze_message(
                message=target_message["text"],
                claimed_sender=target_message["sender_claimed"],
                run_significance=True
            )

        # 1. BOLD VERDICT BANNER
        is_hijack = forensic_res["is_hijack"]
        box_class = "verdict-box-danger" if is_hijack else "verdict-box-success"
        verdict_icon = "🚨" if is_hijack else "✅"
        verdict_headline = "HIJACK DETECTED - UNAUTHORIZED AUTHOR" if is_hijack else "LEGITIMATE AUTHOR VERIFIED"

        st.markdown(f"""
        <div class='{box_class}'>
            <div style='font-size:1.5rem; font-weight:800;'>{verdict_icon} {verdict_headline}</div>
            <div style='font-size:1.2rem; margin-top:6px;'>
                Claimed Account: <b>{target_message['sender_claimed']}</b> | Forensically Identified Typist: <span style='text-decoration:underline;'><b>{forensic_res['verdict']}</b></span>
            </div>
            <div style='font-size:1.1rem; margin-top:6px;'>
                Forensic Confidence: <b>{forensic_res['confidence']}%</b>
            </div>
            <div style='font-size:0.95rem; margin-top:10px; color:#eaeaea;'>
                {forensic_res['reasoning']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. VITERBI DECODED STRING COLORED BY MOOD
        st.markdown("#### 🧬 Viterbi Decoded Keystroke Mood Sequence")
        st.caption("Each edit operation (M: Match, S: Substitution, I: Insertion, D: Deletion) mapped to decoded hidden state.")

        viterbi_states = forensic_res["viterbi_states"]
        ops_str = forensic_res["full_op_string"]

        chip_htmls = []
        for op_char, state in zip(ops_str, viterbi_states):
            bg_color = STATE_COLORS.get(state, "#95a5a6")
            chip_htmls.append(
                f"<span class='viterbi-chip' style='background-color:{bg_color};' title='Mood: {state}'>{op_char}</span>"
            )

        st.markdown("".join(chip_htmls), unsafe_allow_html=True)

        # State legend
        legend_cols = st.columns(3)
        with legend_cols[0]:
            st.markdown(f"<span style='color:{STATE_COLORS['Careful']}; font-weight:bold;'>■ Careful</span> (High Matches)", unsafe_allow_html=True)
        with legend_cols[1]:
            st.markdown(f"<span style='color:{STATE_COLORS['Rushing']}; font-weight:bold;'>■ Rushing</span> (Substitutions & Omissions)", unsafe_allow_html=True)
        with legend_cols[2]:
            st.markdown(f"<span style='color:{STATE_COLORS['Sleepy']}; font-weight:bold;'>■ Sleepy</span> (Key Repeats & Bounces)", unsafe_allow_html=True)

        st.markdown("---")

        # 3. HMM LOG-LIKELIHOOD BAR CHART & PSSM SCORES
        st.markdown("#### 📈 Model Likelihood Comparison")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.caption("HMM Forward Algorithm Log-Likelihood log P(O | λ):")
            hmm_df = pd.DataFrame([
                {"Typist": k, "Log-Likelihood": v}
                for k, v in forensic_res["hmm_scores"].items()
            ])
            fig_hmm, ax_hmm = plt.subplots(figsize=(4.5, 3))
            palette = ["#3498db" if t == target_message["sender_claimed"] else "#e74c3c" for t in hmm_df["Typist"]]
            sns.barplot(data=hmm_df, x="Typist", y="Log-Likelihood", palette=palette, ax=ax_hmm)
            ax_hmm.set_ylabel("Log-Likelihood")
            ax_hmm.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_hmm)
            plt.close(fig_hmm)

        with chart_col2:
            st.caption("PSSM Positional Log-Odds Profile Scores:")
            pssm_df = pd.DataFrame([
                {"Typist": k, "PSSM Score": v}
                for k, v in forensic_res["pssm_scores"].items()
            ])
            fig_pssm, ax_pssm = plt.subplots(figsize=(4.5, 3))
            sns.barplot(data=pssm_df, x="Typist", y="PSSM Score", palette="viridis", ax=ax_pssm)
            ax_pssm.set_ylabel("Log-Odds Score")
            ax_pssm.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_pssm)
            plt.close(fig_pssm)

        # 4. DOT MATRIX HEATMAP (NEEDLEMAN-WUNSCH ALIGNMENT)
        st.markdown("#### 🗺️ Needleman-Wunsch Alignment & Dot Matrix")
        alignments = forensic_res["alignments"]

        if alignments:
            # Dropdown to choose which word pair in the message to visualize
            word_labels = [f"'{a['typed']}' -> '{a['intended']}' (Ops: {a['ops']})" for a in alignments]
            sel_word_idx = st.selectbox("Select Word Alignment to Visualize:", range(len(alignments)), format_func=lambda i: word_labels[i])
            chosen_align = alignments[sel_word_idx]

            st.write(f"**Intended:** `{chosen_align['aligned_intended']}`")
            st.write(f"**Typed:**    `{chosen_align['aligned_typed']}`")
            st.write(f"**Edit Ops:** `{chosen_align['ops']}` | **Gotoh Affine Score:** `{chosen_align['score']:.2f}`")

            # Generate dot matrix
            seq1 = chosen_align["intended"]
            seq2 = chosen_align["typed"]
            dot_mat = generate_dot_matrix(seq1, seq2)

            fig_dot, ax_dot = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                dot_mat,
                annot=True,
                cbar=False,
                cmap="Blues",
                xticklabels=list(seq2),
                yticklabels=list(seq1),
                linewidths=1,
                linecolor="#333",
                ax=ax_dot
            )
            ax_dot.set_xlabel(f"Typed Keystrokes ('{seq2}')")
            ax_dot.set_ylabel(f"Intended Word ('{seq1}')")
            ax_dot.set_title("Dot Plot Keystroke Recurrence Matrix", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_dot)
            plt.close(fig_dot)

        # 5. GUMBEL SIGNIFICANCE
        if forensic_res.get("significance"):
            sig = forensic_res["significance"]
            st.markdown("#### 🎲 Statistical Significance (Gumbel Null Distribution Fit)")
            sig_col1, sig_col2, sig_col3 = st.columns(3)
            sig_col1.metric("Tested Word Pair", f"'{sig['typed_form']}' vs '{sig['tested_word']}'")
            sig_col2.metric("Observed NW Score", f"{sig['real_score']:.2f}")
            sig_col3.metric("Gumbel Fitted p-value", f"{sig['fitted_p_value']:.4e}")
            if sig['fitted_p_value'] < 0.05:
                st.success("✅ The alignment score is statistically significant (p < 0.05). True biological/keystroke homology confirmed.")
            else:
                st.warning("⚠️ High p-value: Alignment could be produced by chance permutation.")


# =====================================================================
# TAB 2: DETECTOR BENCHMARKS & CONFUSION MATRICES
# =====================================================================

with tab_benchmark:
    st.subheader("📊 Four-Model Forensic Detector Comparison")
    st.markdown("""
    Evaluation on held-out test split (360 samples) comparing four bioinformatics detection paradigms:
    1. **Basic Edit Distance Only**: Levenshtein operations heuristic.
    2. **Learned Scoring Matrix Only**: Log-odds substitution scoring matrix without state dynamics.
    3. **HMM Only**: 3-state Baum-Welch Hidden Markov Model on M/S/I/D edit sequences.
    4. **Combined Forensic Toolkit**: Bio-informatic fusion of Gotoh Affine NW + PSSM Log-Odds + HMM.
    """)

    eval_data = get_model_evaluations(engine)
    classes = eval_data["classes"]
    models = eval_data["models"]

    # Accuracies summary cards
    st.markdown("### Model Accuracies")
    acc_cols = st.columns(4)
    for col, (name, m_data) in zip(acc_cols, models.items()):
        acc = m_data["accuracy"] * 100
        col.metric(label=name, value=f"{acc:.1f}%")

    st.markdown("---")
    st.markdown("### 3x3 Confusion Matrices")

    cm_cols1, cm_cols2 = st.columns(2)
    model_items = list(models.items())

    # Plot Matrix 1 & 2
    with cm_cols1:
        for idx in [0, 2]:
            m_name, m_info = model_items[idx]
            st.markdown(f"**{m_name}** (Accuracy: {m_info['accuracy']*100:.1f}%)")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(
                m_info["matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax
            )
            ax.set_xlabel("Predicted Typist")
            ax.set_ylabel("True Typist")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Plot Matrix 3 & 4
    with cm_cols2:
        for idx in [1, 3]:
            m_name, m_info = model_items[idx]
            st.markdown(f"**{m_name}** (Accuracy: {m_info['accuracy']*100:.1f}%)")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(
                m_info["matrix"],
                annot=True,
                fmt="d",
                cmap="Greens" if "Combined" in m_name else "Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax
            )
            ax.set_xlabel("Predicted Typist")
            ax.set_ylabel("True Typist")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Detailed metrics table
    st.markdown("### Forensic Performance Breakdown")
    report_rows = []
    for m_name, m_info in models.items():
        rep = m_info["report"]
        for c in classes:
            report_rows.append({
                "Detector": m_name,
                "Class": c,
                "Precision": f"{rep[c]['precision']:.2f}",
                "Recall": f"{rep[c]['recall']:.2f}",
                "F1-Score": f"{rep[c]['f1-score']:.2f}",
                "Support": int(rep[c]['support'])
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)


# =====================================================================
# TAB 3: INTERACTIVE FORENSIC SANDBOX
# =====================================================================

with tab_sandbox:
    st.subheader("🧪 Live Forensic Keystroke Sandbox")
    st.caption("Test custom messages or word pairs to inspect Gotoh affine alignments, Gumbel shuffle p-values, and Viterbi mood decoding in real time.")

    sandbox_mode = st.radio("Sandbox Mode:", ["Analyze Entire Sentence", "Align Word Pair"], horizontal=True)

    if sandbox_mode == "Analyze Entire Sentence":
        custom_input = st.text_input(
            "Enter custom typed text:",
            value="bro send beriyani mony urgnt akount transfer right now"
        )
        custom_claimed = st.selectbox("Claimed Account Owner:", ["Person_A", "Person_B", "Impostor"])

        if st.button("Run Forensic Analysis"):
            with st.spinner("Executing alignment and Baum-Welch decoding..."):
                res = engine.analyze_message(custom_input, claimed_sender=custom_claimed, run_significance=True)

            st.markdown(f"### Verdict: **{res['verdict']}** ({'🚨 HIJACK' if res['is_hijack'] else '✅ VERIFIED'})")
            st.write(f"**Confidence:** {res['confidence']}%")
            st.info(res['reasoning'])

            st.write(f"**Edit-Operation String:** `{res['full_op_string']}`")
            chips = [
                f"<span class='viterbi-chip' style='background-color:{STATE_COLORS.get(s, '#888')};'>{op}</span>"
                for op, s in zip(res['full_op_string'], res['viterbi_states'])
            ]
            st.markdown("".join(chips), unsafe_allow_html=True)

            st.json({
                "HMM Log-Likelihoods": res["hmm_scores"],
                "PSSM Scores": res["pssm_scores"],
                "Combined Scores": res["combined_scores"]
            })

    else:
        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
            intended_word = st.text_input("Intended Reference Word:", value="tomorrow")
        with sb_col2:
            typed_word = st.text_input("Typed Word (with typos):", value="tmrw")

        gap_open = st.slider("Gap Open Penalty (d):", 0.5, 6.0, 2.5, 0.5)
        gap_extend = st.slider("Gap Extension Penalty (e):", 0.1, 2.0, 0.5, 0.1)

        if st.button("Compute Alignment & Significance"):
            score, a1, a2, ops = needleman_wunsch_affine(
                intended_word, typed_word,
                scoring_matrix=engine.scoring_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend
            )
            sw_score, sw1, sw2, sw_ops, _, _ = smith_waterman_affine(
                intended_word, typed_word,
                scoring_matrix=engine.scoring_matrix,
                gap_open=gap_open,
                gap_extend=gap_extend
            )

            st.markdown("#### Needleman-Wunsch (Global Affine Alignment)")
            st.code(f"Intended: {a1}\nTyped:    {a2}\nOps:      {ops}\nScore:    {score:.2f}")

            st.markdown("#### Smith-Waterman (Local Affine Alignment)")
            st.code(f"Intended: {sw1}\nTyped:    {sw2}\nOps:      {sw_ops}\nScore:    {sw_score:.2f}")

            with st.spinner("Running 100 random shuffles for Gumbel fitting..."):
                sig = compute_alignment_significance(
                    intended_word, typed_word,
                    scoring_matrix=engine.scoring_matrix,
                    n_shuffles=100,
                    gap_open=gap_open,
                    gap_extend=gap_extend
                )
            st.markdown("#### Statistical Significance")
            st.write(f"- **Real Score:** `{sig['real_score']:.2f}`")
            st.write(f"- **Mean Null Score:** `{sig['mean_null']:.2f}` (std: `{sig['std_null']:.2f}`)")
            st.write(f"- **Gumbel Location (μ):** `{sig['gumbel_loc']:.2f}`, **Scale (β):** `{sig['gumbel_scale']:.2f}`")
            st.write(f"- **Empirical p-value:** `{sig['empirical_p_value']:.4f}`")
            st.write(f"- **Fitted Gumbel p-value:** `{sig['fitted_p_value']:.4e}`")

