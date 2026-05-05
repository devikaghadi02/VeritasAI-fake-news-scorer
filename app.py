# app.py — VeritasAI Fake News Credibility Scorer
# Run with: streamlit run app.py

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

from predict import load_roberta, load_meta_clf, predict, extract_article

st.set_page_config(
    page_title="VeritasAI — Fake News Scorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,400;0,500;1,400&family=DM+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main > div { background: #F6F3EE !important; }

[data-testid="stSidebar"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* MASTHEAD */
.mast { width:100%; background:#111; border-bottom:4px solid #C9A84C; }
.mast-inner { max-width:1200px; margin:0 auto; padding:1.6rem 3rem 0; }
.mast-top {
    display:flex; align-items:flex-end; justify-content:space-between;
    padding-bottom:1.2rem; border-bottom:1px solid #282828;
}
.mast-logo {
    font-family:'Playfair Display',serif; font-size:3rem; font-weight:900;
    color:#F5F1EB; letter-spacing:-1.5px; line-height:1;
}
.mast-logo em { color:#C9A84C; font-style:normal; }
.mast-tagline { font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#666; margin-top:6px; font-style:italic; }
.mast-date { font-family:'DM Mono',monospace; font-size:0.72rem; color:#555; }
.mast-nav {
    display:flex; gap:2.5rem; padding:0.8rem 0;
    font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:2px;
    text-transform:uppercase; color:#666;
}
.mast-nav .active { color:#C9A84C; }

/* PAGE BODY */
.page-body { max-width:1200px; margin:0 auto; padding:3rem 3rem 4rem; }

.rule-heavy { border:none; border-top:3px solid #111; margin:0 0 2rem; }
.rule-light  { border:none; border-top:1px solid #DDD8CE; margin:2.5rem 0; }

.page-hed {
    font-family:'Playfair Display',serif; font-size:3rem; font-weight:700;
    color:#111; line-height:1.2; margin-bottom:0.6rem;
}
.page-dek {
    font-family:'DM Sans',sans-serif; font-size:1.1rem; color:#666;
    font-style:italic; line-height:1.7; margin-bottom:2.5rem; max-width:680px;
}
.input-eyebrow {
    font-family:'DM Mono',monospace; font-size:0.65rem;
    letter-spacing:2.5px; text-transform:uppercase; color:#AAA; margin-bottom:0.6rem;
}

/* Streamlit widget overrides */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:transparent !important; border-bottom:2px solid #DDD8CE !important; gap:0 !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'DM Mono',monospace !important; font-size:0.72rem !important;
    letter-spacing:1.5px !important; text-transform:uppercase !important;
    color:#999 !important; background:transparent !important;
    border-radius:0 !important; padding:0.6rem 1.4rem !important; border:none !important;
}
div[data-testid="stTabs"] [aria-selected="true"] { background:#111 !important; color:#F5F1EB !important; }

div[data-testid="stTextArea"] textarea {
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    background:#fff !important; border:1.5px solid #D5CEBC !important;
    border-radius:0 !important; color:#111 !important; line-height:1.65 !important; padding:1rem !important;
}
div[data-testid="stTextInput"] input {
    font-family:'DM Mono',monospace !important; font-size:0.88rem !important;
    background:#fff !important; border:1.5px solid #D5CEBC !important;
    border-radius:0 !important; color:#111 !important; padding:0.75rem 1rem !important;
}
div[data-testid="stButton"] button {
    font-family:'DM Mono',monospace !important; font-size:0.8rem !important;
    letter-spacing:2px !important; text-transform:uppercase !important;
    background:#111 !important; color:#F5F1EB !important; border:none !important;
    border-radius:0 !important; padding:0.85rem 2.5rem !important;
    border-bottom:3px solid #C9A84C !important;
}
div[data-testid="stButton"] button:hover { background:#2A2A2A !important; }

/* VERDICT */
.verdict-card {
    border-left:6px solid; padding:2rem 2.5rem; margin:2rem 0;
    display:flex; align-items:center; justify-content:space-between;
}
.verdict-eyebrow {
    font-family:'DM Mono',monospace; font-size:0.65rem;
    letter-spacing:3px; text-transform:uppercase; opacity:0.6; margin-bottom:8px;
}
.verdict-name { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:900; line-height:1; }
.verdict-num  { font-family:'Playfair Display',serif; font-size:5rem; font-weight:900; line-height:1; text-align:right; }
.verdict-num-label {
    font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:2px;
    text-transform:uppercase; opacity:0.55; margin-top:4px; text-align:right;
}

/* SIGNAL CARDS */
.sig-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:2rem; }
.sig-card { background:#fff; border:1px solid #DDD8CE; border-top:4px solid; padding:1.4rem 1.6rem; }
.sig-eyebrow { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:2px; text-transform:uppercase; color:#BBB; margin-bottom:4px; }
.sig-name { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:#111; margin-bottom:12px; line-height:1.3; }
.sig-score { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:900; line-height:1; margin-bottom:8px; }
.sig-denom { font-size:1.1rem; color:#CCC; }
.sig-bar-bg { height:4px; background:#F0EDE6; margin-bottom:10px; }
.sig-bar-fill { height:100%; }
.sig-detail { font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#AAA; line-height:1.5; }

/* SECTION */
.sec-label {
    font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:3px;
    text-transform:uppercase; color:#AAA; border-bottom:1px solid #DDD8CE;
    padding-bottom:0.6rem; margin-bottom:1.2rem;
}
.sec-title { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; color:#111; margin-bottom:1.1rem; }

/* FINDINGS */
.finding {
    display:flex; gap:12px; padding:0.9rem 0; border-bottom:1px solid #F0EDE6;
    font-family:'DM Sans',sans-serif; font-size:0.97rem; color:#333;
    line-height:1.55; align-items:flex-start;
}
.finding:last-child { border-bottom:none; }

/* STYLE TABLE */
.sf-row { display:flex; align-items:center; padding:0.55rem 0; border-bottom:1px solid #F5F2EC; font-family:'DM Sans',sans-serif; font-size:0.92rem; color:#444; }
.sf-row:last-child { border-bottom:none; }
.sf-name { flex:1; }
.sf-val { font-family:'DM Mono',monospace; font-size:0.8rem; color:#999; min-width:62px; text-align:right; margin-right:12px; }
.sf-flag { min-width:20px; font-size:0.85rem; }

.disclaimer { font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#BBB; font-style:italic; border-top:1px solid #DDD8CE; padding-top:1.2rem; margin-top:1rem; line-height:1.7; }
.footer { background:#111; color:#555; text-align:center; padding:1.3rem; margin-top:3rem; font-family:'DM Mono',monospace; font-size:0.65rem; }
.footer em { color:#C9A84C; font-style:normal; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def verdict_color(score):
    if score >= 65: return "#EDF7EE", "#2D7A3A", "#1D5228"
    if score >= 35: return "#FDF6E3", "#A07820", "#6B4F10"
    return "#FDECEA", "#B03020", "#7A1E12"

def signal_color(v):
    if v >= 65: return "#2D7A3A"
    if v >= 35: return "#A07820"
    return "#B03020"


# ── MASTHEAD ──────────────────────────────────────────────────────────────────
today = datetime.now().strftime("%A, %b %d, %Y").upper()
st.markdown(f"""
<div class="mast">
  <div class="mast-inner">
    <div class="mast-top">
      <div>
        <div class="mast-logo">Veritas<em>AI</em></div>
        <div class="mast-tagline">News credibility analysis</div>
      </div>
      <div class="mast-date">{today}</div>
    </div>
    <div class="mast-nav">
      <span class="active">Credibility Checker</span>
      <span>Source Index</span>
      <span>Methodology</span>
      <span>About</span>
    </div>
  </div>
</div>
<div class="page-body">
""", unsafe_allow_html=True)


# ── MODELS ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    roberta_model, tokenizer = load_roberta()
    meta_clf = load_meta_clf()
    return roberta_model, tokenizer, meta_clf

with st.spinner("Loading AI models…"):
    roberta_model, tokenizer, meta_clf = get_models()


# ── HEADLINE ──────────────────────────────────────────────────────────────────
st.markdown('<hr class="rule-heavy">', unsafe_allow_html=True)
st.markdown("""
<div class="page-hed">Is this article credible?</div>
<div class="page-dek">
  Paste article text or enter a URL. Three independent signals — language model,
  writing-style analysis, and source reputation — return a credibility score in seconds.
</div>
""", unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-eyebrow">Submit an article for analysis</div>', unsafe_allow_html=True)

input_tab, url_tab = st.tabs(["  Paste text  ", "  Enter URL  "])
article_text = ""
article_url  = ""

with input_tab:
    raw = st.text_area("", height=180,
                        placeholder="Paste the full article body or a news statement here…",
                        label_visibility="collapsed")
    if raw:
        article_text = raw
        article_url  = ""

with url_tab:
    url_input = st.text_input("", placeholder="https://www.example.com/news/article-title",
                               label_visibility="collapsed")
    if url_input:
        with st.spinner("Fetching article…"):
            extracted = extract_article(url_input)
        if extracted["success"]:
            article_text = extracted["text"]
            article_url  = extracted["source_url"]
            st.success(f"Extracted: {extracted['title']}")
            with st.expander("Preview extracted text"):
                st.write(article_text[:2000] + ("…" if len(article_text) > 2000 else ""))
        else:
            st.error(f"Could not extract: {extracted['error']}")

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyse Credibility →")


# ── RESULTS ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not article_text.strip():
        st.warning("Please paste some text or enter a valid URL first.")
    elif len(article_text.split()) < 10:
        st.warning("Text is too short — please provide at least 10 words.")
    else:
        with st.spinner("Running three-signal analysis…"):
            result = predict(article_text, article_url, roberta_model, tokenizer, meta_clf)

        score  = result["credibility_score"]
        lname  = result["label_name"]
        lemoji = result["label_emoji"]
        bg, bord, txt = verdict_color(score)

        st.markdown('<hr class="rule-light">', unsafe_allow_html=True)

        # ── Verdict ───────────────────────────────────────────────────────────
        st.markdown(
            f'<div class="verdict-card" style="background:{bg};border-left-color:{bord};">'
            f'  <div>'
            f'    <div class="verdict-eyebrow" style="color:{txt};">Credibility verdict</div>'
            f'    <div class="verdict-name" style="color:{txt};">{lemoji}&nbsp;{lname}</div>'
            f'  </div>'
            f'  <div>'
            f'    <div class="verdict-num" style="color:{bord};">{score:.0f}</div>'
            f'    <div class="verdict-num-label" style="color:{txt};">Score / 100</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Signal cards — built as one string, ONE st.markdown call ──────────
        sb  = result["signal_breakdown"]
        rb  = sb["RoBERTa (text analysis)"]
        sty = sb["Writing style"]["style_score"]
        src = sb["Source reputation"]["score"] if article_url else 50.0

        def sig_card_html(eyebrow, name, val, detail):
            c = signal_color(val)
            bar_w = f"{val:.1f}"
            return (
                f'<div class="sig-card" style="border-top-color:{c};">'
                f'  <div class="sig-eyebrow">{eyebrow}</div>'
                f'  <div class="sig-name">{name}</div>'
                f'  <div class="sig-score" style="color:{c};">{val:.0f}'
                f'    <span class="sig-denom">/100</span></div>'
                f'  <div class="sig-bar-bg">'
                f'    <div class="sig-bar-fill" style="width:{bar_w}%;background:{c};"></div>'
                f'  </div>'
                f'  <div class="sig-detail">{detail}</div>'
                f'</div>'
            )

        src_label = (
            f"Domain trust score: {src:.0f}/100"
            if article_url
            else "No URL provided — neutral 50/100 used"
        )

        grid_html = (
            '<div class="sig-grid">'
            + sig_card_html(
                "Signal 01 &middot; Language model", "RoBERTa text analysis",
                rb["real"],
                f"Real {rb['real']:.0f}% &nbsp;&middot;&nbsp; Mixed {rb['mixed']:.0f}% &nbsp;&middot;&nbsp; Fake {rb['fake']:.0f}%",
            )
            + sig_card_html(
                "Signal 02 &middot; Writing style", "Stylometric analysis",
                sty,
                "Caps, exclamations, hedging language, emotional words, vocabulary richness",
            )
            + sig_card_html(
                "Signal 03 &middot; Source trust", "Domain reputation",
                src, src_label,
            )
            + '</div>'
        )
        st.markdown(grid_html, unsafe_allow_html=True)

        st.markdown('<hr class="rule-light">', unsafe_allow_html=True)

        # ── Findings + style features ─────────────────────────────────────────
        left, right = st.columns([1.2, 0.8])
        sf = result["style_feats"]

        with left:
            st.markdown('<div class="sec-title">Key findings</div>', unsafe_allow_html=True)
            findings = []
            if sf[5] > 0.5:
                findings.append(("🚩", "High emotional language — a common marker of misleading content."))
            if sf[1] > 0.1:
                findings.append(("🚩", "Excessive ALL CAPS usage — aggressive tone signalling."))
            if sf[0] > 0.02:
                findings.append(("🚩", "Above-average exclamation use — characteristic of tabloid writing."))
            if sf[4] > 1.0:
                findings.append(("✅", "Uses hedging language ('reportedly', 'allegedly') — consistent with responsible journalism."))
            if sf[6] > 0.6:
                findings.append(("✅", "Rich, varied vocabulary — associated with professionally edited writing."))
            if result["source_score"] > 0.85:
                findings.append(("✅", f"Source carries a high trust score of {result['source_score']*100:.0f}/100."))
            if result["source_score"] < 0.25 and article_url:
                findings.append(("🚩", f"Source domain has a low trust score of {result['source_score']*100:.0f}/100."))
            if not findings:
                findings.append(("ℹ️", "No strongly positive or negative style signals detected."))

            findings_html = "".join(
                f'<div class="finding">'
                f'  <span style="flex-shrink:0;margin-top:2px;">{ic}</span>'
                f'  <span>{tx}</span>'
                f'</div>'
                for ic, tx in findings
            )
            st.markdown(findings_html, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="sec-title">Style features</div>', unsafe_allow_html=True)
            rows = [
                ("Exclamation ratio",   sf[0], True),
                ("ALL CAPS ratio",      sf[1], True),
                ("Avg word length",     sf[2], None),
                ("Question ratio",      sf[3], True),
                ("Hedging score",       sf[4], False),
                ("Emotional language",  sf[5], True),
                ("Vocabulary richness", sf[6], False),
                ("Avg sentence len",    sf[7], None),
            ]

            def flag(v, h):
                if h is None: return ""
                return "🚩" if (h and v > 0.5) or (not h and v < 0.3) else "✓"

            rows_html = "".join(
                f'<div class="sf-row">'
                f'  <span class="sf-name">{n}</span>'
                f'  <span class="sf-val">{v:.3f}</span>'
                f'  <span class="sf-flag">{flag(v, h)}</span>'
                f'</div>'
                for n, v, h in rows
            )
            st.markdown(rows_html, unsafe_allow_html=True)

        st.markdown('<hr class="rule-light">', unsafe_allow_html=True)

        # ── Probability chart ─────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Final classifier probabilities</div>', unsafe_allow_html=True)
        probs = result["meta_probs"]
        vals  = [probs["real"], probs["mixed"], probs["fake"]]

        fig = go.Figure(go.Bar(
            x=["Likely Real", "Mixed / Uncertain", "Likely Fake"],
            y=vals,
            marker_color=["#2D7A3A", "#A07820", "#B03020"],
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(family="DM Mono", size=13, color="#555"),
            width=0.35,
        ))
        fig.update_layout(
            height=280,
            margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(
                range=[0, max(vals) * 1.4],
                showgrid=True, gridcolor="#EDEBE5",
                ticksuffix="%",
                tickfont=dict(family="DM Mono", size=11, color="#BBB"),
            ),
            xaxis=dict(tickfont=dict(family="DM Sans", size=14, color="#444")),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="disclaimer">VeritasAI is an AI-assisted tool. Results are probabilistic '
            'estimates, not editorial verdicts. Always verify with primary sources and '
            'independent fact-checkers.</div>',
            unsafe_allow_html=True,
        )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    '</div>'
    '<div class="footer">'
    '  <em>VeritasAI</em> &nbsp;&middot;&nbsp; '
    '  Built with RoBERTa &middot; Gradient Boosting &middot; Streamlit'
    '  &nbsp;&middot;&nbsp; Trained on the LIAR dataset (PolitiFact)'
    '</div>',
    unsafe_allow_html=True,
)