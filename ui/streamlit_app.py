"""
Phase 7 — Streamlit UI
Colour-coded hallucination detection interface with:
- Sentence-level verdict highlighting
- Trust score meter
- Claim-by-claim sidebar breakdown
- Side-by-side original vs rewritten view
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/check"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.supported {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    color: #155724;
}
.contradicted {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    color: #721c24;
}
.unverifiable {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    color: #856404;
}
.non-factual {
    background-color: #f8f9fa;
    border-left: 4px solid #6c757d;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    color: #495057;
}
.rewritten {
    background-color: #e8f4fd;
    border-left: 4px solid #0d6efd;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    color: #084298;
    font-style: italic;
}
.trust-high { color: #28a745; font-size: 2rem; font-weight: bold; }
.trust-mid  { color: #ffc107; font-size: 2rem; font-weight: bold; }
.trust-low  { color: #dc3545; font-size: 2rem; font-weight: bold; }
.verdict-badge {
    font-size: 0.7rem;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────

ICONS = {
    "supported":    "✅",
    "contradicted": "❌",
    "unverifiable": "❓",
    "non-factual":  "⬜",
}

LABELS = {
    "supported":    "SUPPORTED",
    "contradicted": "HALLUCINATED",
    "unverifiable": "UNCERTAIN",
    "non-factual":  "NON-FACTUAL",
}


def trust_color_class(score: float) -> str:
    if score >= 0.7:
        return "trust-high"
    elif score >= 0.4:
        return "trust-mid"
    return "trust-low"


def call_api(text: str, rewrite: bool) -> dict | None:
    try:
        resp = requests.post(
            API_URL,
            json={"text": text, "rewrite_hallucinations": rewrite},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure `uvicorn api.main:app --reload` is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The pipeline is still processing — try again or use shorter text.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Hallucination Detector")
    st.caption("Real-time LLM fact verification")
    st.divider()

    st.subheader("⚙️ Options")
    rewrite_mode = st.toggle("✏️ Auto-rewrite hallucinations", value=False)

    st.divider()
    st.subheader("📖 Legend")
    st.markdown("✅ **Green** — Verified by web sources")
    st.markdown("❌ **Red** — Contradicted by web sources")
    st.markdown("❓ **Amber** — Insufficient evidence")
    st.markdown("⬜ **Grey** — Not a factual claim")

    st.divider()
    st.subheader("💡 Try these examples")

    example1 = "Albert Einstein was born in Germany in 1879. He invented the telephone. He won the Nobel Prize in Physics in 1921."
    example2 = "The Great Wall of China is visible from space. It was built during the Ming Dynasty. It stretches over 13,000 miles."
    example3 = "Python was created by Guido van Rossum. The first version was released in 1991. Python is named after Monty Python."

    if st.button("🧪 Einstein example"):
        st.session_state["input_text"] = example1
    if st.button("🧱 Great Wall example"):
        st.session_state["input_text"] = example2
    if st.button("🐍 Python example"):
        st.session_state["input_text"] = example3

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🔍 Real-Time Hallucination Detector")
st.caption("Paste any LLM-generated text below. Each sentence will be verified against live web sources.")

# Text input
default_text = st.session_state.get("input_text", "")
input_text = st.text_area(
    "LLM Response to verify:",
    value=default_text,
    height=150,
    placeholder="Paste any AI-generated text here...",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("🔍 Verify", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=False):
        st.session_state["input_text"] = ""
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────

if run_button and input_text.strip():
    with st.spinner("🔍 Verifying claims against live web sources... (this takes 20–40 seconds)"):
        result = call_api(input_text.strip(), rewrite_mode)

    if result:
        annotations = result["annotations"]
        trust_score = result["overall_trust_score"]
        trust_pct = int(trust_score * 100)

        st.divider()

        # ── Trust Score ──────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Trust Score", f"{trust_pct}%")
        m2.metric("✅ Verified",     result["verified_count"])
        m3.metric("❌ Hallucinated", result["hallucination_count"])
        m4.metric("❓ Uncertain",    result["unverifiable_count"])

        st.progress(trust_score)
        st.divider()

        # ── Colour-coded annotations ─────────────────────────────────────────
        if rewrite_mode and any(a.get("rewritten") for a in annotations):
            left_col, right_col = st.columns(2)
            with left_col:
                st.subheader("📄 Original")
            with right_col:
                st.subheader("✏️ Rewritten")
        else:
            st.subheader("📄 Annotated Response")

        for ann in annotations:
            verdict  = ann["verdict"]
            sentence = ann["sentence"]
            icon     = ICONS.get(verdict, "?")
            label    = LABELS.get(verdict, verdict.upper())
            conf     = int(ann["confidence"] * 100)
            rewritten = ann.get("rewritten")

            if rewrite_mode and any(a.get("rewritten") for a in annotations):
                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown(
                        f'<div class="{verdict}">{icon} <strong>[{label}]</strong> '
                        f'{sentence} <small>({conf}% confidence)</small></div>',
                        unsafe_allow_html=True,
                    )
                with right_col:
                    if rewritten:
                        st.markdown(
                            f'<div class="rewritten">✏️ {rewritten}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="{verdict}">{icon} {sentence}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    f'<div class="{verdict}">{icon} <strong>[{label}]</strong> '
                    f'{sentence} <small>({conf}% confidence)</small></div>',
                    unsafe_allow_html=True,
                )

        # ── Claim breakdown (expander) ────────────────────────────────────────
        st.divider()
        st.subheader("🔬 Claim-by-Claim Breakdown")

        for ann in annotations:
            verdict  = ann["verdict"]
            icon     = ICONS.get(verdict, "?")
            sentence = ann["sentence"]

            with st.expander(f"{icon} {sentence[:80]}{'...' if len(sentence) > 80 else ''}"):
                st.markdown(f"**Sentence verdict:** `{verdict.upper()}` ({int(ann['confidence']*100)}% confidence)")

                if ann.get("sources"):
                    st.markdown("**Sources:**")
                    for url in ann["sources"][:3]:
                        st.markdown(f"- [{url}]({url})")

                if ann.get("claims"):
                    st.markdown("**Extracted claims:**")
                    for claim in ann["claims"]:
                        claim_icon = ICONS.get(claim["verdict"], "?")
                        st.markdown(
                            f"{claim_icon} `{claim['verdict'].upper()}` — {claim['claim']}"
                        )
                        if claim.get("evidence"):
                            top = claim["evidence"][0]
                            st.caption(
                                f"Top evidence (similarity: {top['similarity_score']:.2f}): "
                                f"{top['content'][:200]}..."
                            )

        # ── Raw JSON (collapsible) ────────────────────────────────────────────
        with st.expander("🗂️ Raw API Response (JSON)"):
            st.json(result)

elif run_button and not input_text.strip():
    st.warning("Please enter some text to verify.")