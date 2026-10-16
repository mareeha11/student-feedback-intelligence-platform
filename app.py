import io
import time
import hashlib
from collections import Counter
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from engine import initialize_deep_learning_backbone, predict
from summarizer_engine import build_dataset_report, build_single_report

# =============================================================================
# 1. CORE CONFIGURATIONS (MUST BE DECLARED FIRST)
# =============================================================================

APP_TITLE = "Feedback Intelligence"
PILLAR_ORDER = ["Assessment", "Course Structure", "Lecturer", "Subject Material", "Other"]
VALID_SENTIMENTS = ["Positive", "Neutral", "Negative"]

SENTIMENT_COLORS = {
    "Positive": "#10b981",
    "Neutral": "#6b7280",
    "Negative": "#ef4444",
}

PILLAR_ICONS = {
    "Assessment": "📝",
    "Course Structure": "📐",
    "Lecturer": "👨‍🏫",
    "Subject Material": "📚",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. STREAMLINED CSS STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: #0a0a0b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e5e5e5;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    h1, h2, h3, h4 { 
        color: #fafafa !important; 
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .metric {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #71717a;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #fafafa;
    }
    
    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-positive { background: #064e3b; color: #34d399; }
    .badge-negative { background: #7f1d1d; color: #fca5a5; }
    .badge-neutral { background: #27272a; color: #a1a1aa; }
    .badge-info { background: #1e3a5f; color: #7dd3fc; }
    
    .stButton > button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.625rem 1.25rem !important;
        transition: all 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover { background: #2563eb !important; transform: translateY(-1px); }
    
    .stDownloadButton > button { background: #18181b !important; border: 1px solid #3f3f46 !important; color: #fafafa !important; width: 100%; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #18181b; border-radius: 8px; padding: 0.5rem 1rem; color: #a1a1aa; }
    .stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; }
    
    .stTextArea textarea { background: #18181b !important; border: 1px solid #27272a !important; border-radius: 8px !important; color: #e5e5e5 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stProgress > div > div { background: #3b82f6 !important; }
    hr { border: none; border-top: 1px solid #27272a; margin: 1.5rem 0; }
    .empty-state { text-align: center; padding: 4rem 2rem; color: #71717a; }
    
    section[data-testid="stSidebar"] { background: #0f0f10; border-right: 1px solid #27272a; }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid #27272a !important;
        color: #a1a1aa !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { background: #18181b !important; color: #fafafa !important; border-color: #3f3f46 !important; }
    
    .result-card { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; }
    .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #27272a; }
    .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .result-item { padding: 0.75rem; background: #0f0f10; border-radius: 8px; }
    .result-item-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717a; margin-bottom: 0.25rem; }
    .result-item-value { color: #fafafa; font-weight: 500; font-size: 0.95rem; }
    
    .priority-high { border-left: 3px solid #ef4444; }
    .priority-medium { border-left: 3px solid #f59e0b; }
    .priority-low { border-left: 3px solid #10b981; }
    .keyword-tag { background: #1e3a5f; color: #7dd3fc; padding: 2px 8px; border-radius: 4px; font-weight: 500; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. HELPER FUNCTIONS & FORMAT PARSERS (ORDER ALIGNED FOR VALIDATION)
# =============================================================================

def short_feedback(text, max_chars=140):
    """Clean and truncate string lines safely to protect boundaries."""
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars: return clean
    return clean[:max_chars - 3].rstrip() + "..."

def normalize_sentiment(sentiment: str) -> str:
    sentiment = str(sentiment).strip().title()
    if sentiment in VALID_SENTIMENTS: return sentiment
    if sentiment.lower() in ["pos", "positive", "good"]: return "Positive"
    if sentiment.lower() in ["neg", "negative", "bad"]: return "Negative"
    return "Neutral"

def render_metric(label: str, value: str, style: str = ""):
    style_class = f"metric-value {style}" if style else "metric-value"
    return f"""<div class="metric"><div class="metric-label">{escape(label)}</div><div class="{style_class}">{escape(value)}</div></div>"""

def get_sentiment_badge(sentiment: str) -> str:
    sentiment = normalize_sentiment(sentiment)
    variant = "positive" if sentiment == "Positive" else ("negative" if sentiment == "Negative" else "neutral")
    return f'<span class="badge badge-{variant}">{sentiment}</span>'

def get_priority_class(sentiment: str, has_suggestion: bool) -> tuple[str, str]:
    sentiment = normalize_sentiment(sentiment)
    if sentiment == "Negative" and has_suggestion: return "priority-high", "High Priority"
    elif sentiment == "Negative" or has_suggestion: return "priority-medium", "Medium Priority"
    return "priority-low", "Low Priority"

def parse_split_single_report(raw_text: str) -> dict:
    parsed = {"Summary": "", "Recommendation": ""}
    current_key = None
    for line in str(raw_text).splitlines():
        line_strip = line.strip()
        if line_strip.lower().startswith("summary:"):
            current_key = "Summary"
            parsed["Summary"] = line_strip.split(":", 1)[1].strip()
        elif line_strip.lower().startswith("recommendation:") or line_strip.lower().startswith("recommendations:"):
            current_key = "Recommendation"
            parsed["Recommendation"] = line_strip.split(":", 1)[1].strip()
        elif current_key and line_strip:
            parsed[current_key] += " " + line_strip
    return parsed

def format_report_html(raw_text: str) -> str:
    """
    Ironclad multi-line layout engine. Intercepts split lines, cleans out stray 
    characters, and dynamically merges fragmented LLM segments into clean cards.
    """
    import re
    
    # Step 1: Scrub out raw HTML remnants or leaked tags
    text_clean = str(raw_text)
    text_clean = re.sub(r'<div[^>]*>', '', text_clean)
    text_clean = text_clean.replace('</div>', '').replace('</span>', '').replace('<span[^>]*>', '')
    
    # Pre-collapse any separate rows that contain loose structural bars
    text_clean = re.sub(r'\n\s*\|\s*\n', ' ', text_clean)
    text_clean = re.sub(r'\n\s*\|\s*', ' ', text_clean)
    
    # Step 2: Extract and structuralize the Priority Action Matrix safely
    if "# PRIORITY ACTION MATRIX" in text_clean:
        main_parts = text_clean.split("# PRIORITY ACTION MATRIX", 1)
        body_content = main_parts[0]
        matrix_content = main_parts[1]
        
        # Regex scans for any block starting with a digit list marker (e.g., 1., 2.)
        matrix_items = re.findall(r'(\d+[\.\s][^\n]+(?:(?!\n\d+[\.\s]|\n#).)*)', matrix_content, re.DOTALL)
        formatted_matrix_elements = ["<h3 style='color:#f59e0b; border-bottom:1px solid #27272a; padding-bottom:6px; margin-top:1.25rem; font-size:15px; text-transform:uppercase; letter-spacing:0.5px;'># PRIORITY ACTION MATRIX</h3>"]
        
        for item in matrix_items:
            # Breakdown the item block lines cleanly
            item_strip = item.strip()
            item_lines = [line.strip() for line in item_strip.split("\n") if line.strip()]
            
            if not item_lines: continue
            
            # Line 1 is always our primary action
            primary_line = item_lines[0].replace("**", "").strip()
            
            # Check if line 2 or 3 holds our trailing target impact statement
            impact_text = ""
            if len(item_lines) > 1:
                # Catch the second line text, stripping out redundant emoji characters
                impact_text = item_lines[1].replace("**", "").replace("🎯", "").replace("|", "").strip()
                # If line 2 was just a stray pipe character, check line 3
                if (impact_text == "" or impact_text == "|") and len(item_lines) > 2:
                    impact_text = item_lines[2].replace("**", "").replace("🎯", "").replace("|", "").strip()
            
            # Strict Split: Isolate strictly by the em-dash delimiter generated by the prompt
            if " — " in primary_line:
                parts = primary_line.split(" — ")
                num_and_action = parts[0].strip()
                # Safely pull impact text; handle edge cases where the LLM might append trailing data
                raw_impact = parts[2].strip() if len(parts) > 2 else (parts[1].strip() if len(parts) > 1 else "Pedagogical Optimization Strategy")
                impact_text = raw_impact.replace("🎯", "").strip()
            else:
                num_and_action = primary_line
                impact_text = "Pedagogical Optimization Strategy"
            
            # Strip trailing garbage symbols from our action statement string ends
            if num_and_action.endswith(" |") or num_and_action.endswith(" —") or num_and_action.endswith(" -") or num_and_action.endswith("|"):
                num_and_action = re.sub(r'[\s\|\-\—]+$', '', num_and_action).strip()
            
            # Ensure an impact string fallback is available
            if not impact_text or impact_text == "|":
                impact_text = "Pedagogical Optimization Strategy"
                
            # Render our final high-contrast blue container safely
            formatted_matrix_elements.append(f"""
            <div style="background:#111827; padding:0.75rem 1rem; border-radius:8px; border-left:4px solid #3b82f6; margin-bottom:0.6rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);">
                <div style="font-size:13px; font-weight:600; color:#fafafa; margin-bottom:0.2rem;">{num_and_action}</div>
                <div style="display:flex; gap:0.5rem; font-size:11px;">
                    <span style="color:#38bdf8; font-weight:700; text-transform:uppercase;">📍 Focus Axis</span>
                    <span style="color:#94a3b8;">|</span>
                    <span style="color:#10b981; font-weight:500;">🎯 {impact_text}</span>
                </div>
            </div>
            """)
            
        text_clean = body_content + "\n".join(formatted_matrix_elements)

    # Step 3: Run standard line parsing over the rest of the text
    lines = text_clean.strip().split("\n")
    html_parts = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        # Guard clause: bypass rows we already formatted into custom containers
        if 'margin-bottom:0.6rem;' in stripped or 'text-transform:uppercase;' in stripped or stripped == '|' or stripped.startswith('🎯'):
            if 'background:#111827;' in stripped or 'display:flex;' in stripped or 'h3' in stripped:
                html_parts.append(stripped)
            continue
            
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            html_parts.append(f'<h3 style="color:#f59e0b; border-bottom:1px solid #27272a; padding-bottom:6px; margin-top:1.25rem; font-size:15px; text-transform:uppercase; letter-spacing:0.5px;">{title}</h3>')
        elif stripped.startswith("## "):
            title = stripped[3:].strip()
            html_parts.append(f'<h4 style="color:#3b82f6; margin-top:1rem; margin-bottom:0.4rem; font-size:14px; font-weight:600;">{title}</h4>')
        elif stripped.startswith("**") and "**:" in stripped:
            parts = stripped.split("**:", 1)
            label = parts[0].replace("**", "").strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            html_parts.append(f'<p style="margin:0.4rem 0; font-size:13px;"><strong style="color:#38bdf8;">{label}:</strong> <span style="color:#e2e8f0;">{content}</span></p>')
        elif stripped.startswith("● ") or stripped.startswith("- IMMEDIATE") or stripped.startswith("- SHORT-TERM") or stripped.startswith("- LONG-TERM"):
            content = stripped.lstrip("●- ").strip()
            if content.upper().startswith("IMMEDIATE"):
                html_parts.append(f'<p style="margin-left:0.5rem; color:#ef4444; font-size:12.5px; font-weight:600; margin-bottom:0.25rem;">🚨 {content}</p>')
            elif content.upper().startswith("SHORT-TERM"):
                html_parts.append(f'<p style="margin-left:0.5rem; color:#f59e0b; font-size:12.5px; font-weight:600; margin-bottom:0.25rem;">⏳ {content}</p>')
            elif content.upper().startswith("LONG-TERM"):
                html_parts.append(f'<p style="margin-left:0.5rem; color:#10b981; font-size:12.5px; font-weight:600; margin-bottom:0.25rem;">🎯 {content}</p>')
            else:
                html_parts.append(f'<p style="margin-left:0.75rem; color:#e2e8f0; font-size:12.5px; margin-bottom:0.25rem;">&bull; {content}</p>')
        elif stripped.startswith("- ") or stripped.startswith("• "):
            content = stripped[2:].strip()
            html_parts.append(f'<p style="margin-left:0.75rem; color:#cbd5e1; font-size:12.5px; margin-bottom:0.25rem; line-height:1.4;">&bull; {content}</p>')
        else:
            html_parts.append(f'<p style="color:#cbd5e1; line-height:1.5; margin:0.4rem 0; font-size:13px;">{stripped}</p>')
            
    return "".join(html_parts)

def extract_section_blocks(raw_text: str) -> dict:
    sections = {"Executive": "", "Categories": {}, "Matrix": ""}
    current_mode = None
    current_pillar = None
    for line in str(raw_text).splitlines():
        line_strip = line.strip()
        if not line_strip: continue
        if line_strip.startswith("# EXECUTIVE SUMMARY"):
            current_mode = "Executive"
            sections["Executive"] += line + "\n"
        elif line_strip.startswith("# CATEGORY ANALYSIS"):
            current_mode = "Categories"
        elif line_strip.startswith("# PRIORITY ACTION MATRIX"):
            current_mode = "Matrix"
            sections["Matrix"] += line + "\n"
        elif current_mode == "Categories" and line_strip.startswith("## "):
            current_pillar = line_strip.replace("##", "").strip(" []")
            sections["Categories"][current_pillar] = line + "\n"
        elif current_mode == "Categories" and current_pillar:
            sections["Categories"][current_pillar] += line + "\n"
        elif current_mode == "Executive":
            sections["Executive"] += line + "\n"
        elif current_mode == "Matrix":
            sections["Matrix"] += line + "\n"
    return sections

def highlight_keywords(text: str) -> str:
    keywords = {"lecturer", "professor", "teacher", "instructor", "assignment", "assignments", "quiz", "quizzes", "exam", "exams", "test", "tests", "grading", "rubric", "structure", "syllabus", "timeline", "schedule", "organized", "material", "content", "slides", "textbook", "exercises", "coding"}
    words = text.split()
    result = []
    for word in words:
        clean_word = word.lower().strip(".,!?\"'()[]:;")
        if clean_word in keywords: result.append(f'<span class="keyword-tag">{escape(word)}</span>')
        else: result.append(escape(word))
    return " ".join(result)

def build_structured_payload(cache: dict) -> dict:
    pillar_data = {}
    for pillar in PILLAR_ORDER:
        if cache["pillar_counts"].get(pillar, 0) == 0: continue
        examples = cache["pillar_examples"].get(pillar, {})
        sentiments = cache["pillar_sentiments"].get(pillar, {})
        pillar_data[pillar] = {
            "count": cache["pillar_counts"][pillar],
            "positive": sentiments.get("Positive", 0),
            "neutral": sentiments.get("Neutral", 0),
            "negative": sentiments.get("Negative", 0),
            "suggestions": cache["pillar_suggestions"].get(pillar, 0),
            "positive_examples": [short_feedback(fb, 200) for fb in examples.get("Positive", [])[:3]],
            "negative_examples": [short_feedback(fb, 200) for fb in examples.get("Negative", [])[:4]],
            "suggestion_examples": [short_feedback(fb, 200) for fb in examples.get("Suggestions", [])[:3]],
        }
    return {
        "total_records": cache["total"], "positive_pct": cache["pos_pct"], "negative_pct": cache["neg_pct"],
        "suggestion_count": cache["sug_count"], "top_pillar": cache["top_pillar"], "risk_pillar": cache["worst_pillar"],
        "pillar_data": pillar_data,
    }

def compile_pdf_report(report_text: str, title: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=14, leading=20, spaceAfter=15)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], alignment=TA_JUSTIFY, fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=8)
    
    story = [Paragraph(escape(title), title_style), Spacer(1, 8)]
    for line in str(report_text).splitlines():
        if line.strip(): story.append(Paragraph(escape(line.strip()), body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =============================================================================
# 4. INITIALIZE SESSION STATES
# =============================================================================

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Analyze Feedback"
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "stats" not in st.session_state:
    st.session_state.stats = {"records": 0, "positive_pct": 0, "negative_pct": 0, "suggestions": 0, "top_pillar": "—", "risk_pillar": "—"}

# =============================================================================
# 5. SIDEBAR MASTER ROUTER
# =============================================================================

with st.sidebar:
    st.markdown("### 🎓 Feedback AI")
    st.caption("Course Quality Monitor")
    st.markdown("---")
    if st.button("🔍  Analyze Feedback", key="nav_Analyze", use_container_width=True): st.session_state.nav_page = "Analyze Feedback"; st.rerun()
    if st.button("📁  Dataset Analysis", key="nav_Dataset", use_container_width=True): st.session_state.nav_page = "Dataset Analysis"; st.rerun()
    if st.button("⚙️  Model Metrics Hub", key="nav_ModelInfo", use_container_width=True): st.session_state.nav_page = "Model Info"; st.rerun()

page = st.session_state.nav_page

# =============================================================================
# 6. ROUTE A: SINGLE FEEDBACK VIEW
# =============================================================================
if page == "Analyze Feedback":
    st.markdown("## Single Feedback Analysis")
    st.caption("Process individual text items to evaluate categorical probability constraints.")
    
    col1, col2, col3 = st.columns(3)
    samples = [
        ("👨‍🏫 Teaching Style Analysis", "The lecturer explains complex machine learning concepts clearly, but additional practical lab sessions would improve understanding."),
        ("📝 Assessment Tracking Criteria", "The continuous assessment requirements are completely unstructured, and grading rubrics were returned too late, causing confusion before final exams."),
        ("📐 Course Structure Metrics", "The module's weekly pacing is heavily front-loaded; transitioning into deep transformer architectures requires introductory tutorials from week one."),
    ]
    for col, (label, sample_text) in zip([col1, col2, col3], samples):
        with col:
            if st.button(label, use_container_width=True): st.session_state.feedback_input = sample_text; st.rerun()
            
    st.markdown("---")
    text = st.text_area("Entry context", value=st.session_state.get("feedback_input", ""), height=100, label_visibility="collapsed", placeholder="Type student feedback here...")
    analyze = st.button("🔍 Analyze Feedback", type="primary", use_container_width=True)
    
    if analyze and text.strip():
        with st.spinner("Analyzing text fields..."):
            result = predict(text)
            result["sentiment"] = normalize_sentiment(result.get("sentiment", "Neutral"))
            report = build_single_report(result, text)
            
        st.markdown("### Evaluation Insights")
        col_l, col_r = st.columns([6, 4])
        with col_l:
            priority_class, priority_label = get_priority_class(result["sentiment"], result.get("suggestion") == "Yes")
            st.markdown(f"""
            <div class="card {priority_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid #27272a; padding-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: #fafafa; font-size: 1rem;">{priority_label} Diagnostic Ledger</span>
                    {get_sentiment_badge(result["sentiment"])}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.5rem;">
                    <div style="padding: 0.75rem; background: #0f0f10; border-radius: 8px;">
                        <div class="result-item-label">Focus Axis Domain</div>
                        <div style="color: #fafafa; font-weight: 500;">{PILLAR_ICONS.get(result['pillars'][0] if result.get('pillars') else 'General', '📋')} {result['pillars'][0] if result.get('pillars') else 'General'}</div>
                    </div>
                    <div style="padding: 0.75rem; background: #0f0f10; border-radius: 8px;">
                        <div class="result-item-label">Suggestion Presence</div>
                        <div style="color: #fafafa; font-weight: 500;">{"✅ Present" if result.get("suggestion") == "Yes" else "❌ None"}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="line-height:2;">{highlight_keywords(text)}</div>', unsafe_allow_html=True)
            
            if report:
                parsed_brief = parse_split_single_report(report)
                if parsed_brief["Summary"] or parsed_brief["Recommendation"]:
                    st.markdown(f"""
                    <div class="card" style="border-left: 3px solid #f59e0b; background:#111827;">
                        <p style="margin-bottom:0.4rem; font-size:13px;"><strong style="color:#f59e0b; font-size:11px; text-transform:uppercase; display:block;">📌 Summary</strong> {parsed_brief['Summary']}</p>
                        <p style="margin-bottom:0px; font-size:13px;"><strong style="color:#38bdf8; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; display:block;">💡 Recommendation</strong> {parsed_brief['Recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else: st.info(report)
        with col_r:
            st.markdown("#### Model Sigmoid Probabilities")
            for cat in PILLAR_ORDER:
                score = result.get("pillar_scores", {}).get(cat, 0.0)
                st.caption(f"{PILLAR_ICONS.get(cat, '📋')} {cat} ({score:.0f}%)")
                st.progress(score / 100.0)

# =============================================================================
# 7. ROUTE B: DATASET ANALYSIS VIEW
# =============================================================================
elif page == "Dataset Analysis":
    st.markdown("## Dataset Analysis & Batch Evaluation")
    file = st.file_uploader("Upload CSV Evaluation Spreadsheet Ingest", type=["csv"], label_visibility="collapsed")
    
    if file:
        file_bytes = file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        if "batch_cache" not in st.session_state or st.session_state.get("cached_file_hash") != file_hash:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
                possible_cols = ["feedback", "text", "comment", "student_feedback", "text_clean"]
                feedback_col = next((c for c in possible_cols if c in df.columns), None)
                if not feedback_col:
                    st.error("No valid text tracking column header row detected.")
                    st.stop()
                    
                progress = st.progress(0)
                rows = df[feedback_col].fillna("").astype(str).tolist()
                total = len(rows)
                
                predictions = []
                pillar_sentiments = {p: {"Positive": 0, "Neutral": 0, "Negative": 0} for p in PILLAR_ORDER}
                pillar_suggestions = {p: 0 for p in PILLAR_ORDER}
                pillar_counts = {p: 0 for p in PILLAR_ORDER}
                pillar_examples = {p: {"Positive": [], "Negative": [], "Neutral": [], "Suggestions": []} for p in PILLAR_ORDER}
                
                for i, text_row in enumerate(rows):
                    progress.progress((i + 1) / total)
                    res = predict(text_row)
                    s_val = normalize_sentiment(res.get("sentiment", "Neutral"))
                    sug_val = res.get("suggestion", "No")
                    assigned = res.get("pillars", []) or ["General"]
                    
                    predictions.append({"Feedback": text_row, "Category": ", ".join(assigned), "Sentiment": s_val, "Suggestion": sug_val})
                    for p in assigned:
                        if p in PILLAR_ORDER:
                            pillar_counts[p] += 1
                            pillar_sentiments[p][s_val] += 1
                            if len(pillar_examples[p][s_val]) < 5: pillar_examples[p][s_val].append(text_row)
                            if sug_val == "Yes":
                                pillar_suggestions[p] += 1
                                if len(pillar_examples[p]["Suggestions"]) < 5: pillar_examples[p]["Suggestions"].append(text_row)
                                
                progress.empty()
                pred_df = pd.DataFrame(predictions)
                
                pos_pct = int((pred_df["Sentiment"] == "Positive").sum() / total * 100) if total > 0 else 0
                neg_pct = int((pred_df["Sentiment"] == "Negative").sum() / total * 100) if total > 0 else 0
                sug_count = int((pred_df["Suggestion"] == "Yes").sum())
                top_pillar = max(pillar_counts, key=pillar_counts.get) if any(pillar_counts.values()) else "None"
                
                ratio_map = {p: (pillar_sentiments[p]["Negative"] / pillar_counts[p] if pillar_counts[p] > 0 else 0) for p in PILLAR_ORDER}
                worst_pillar = max(ratio_map, key=ratio_map.get) if any(ratio_map.values()) else "None"
                
                st.session_state.batch_cache = {
                    "df": pred_df, "total": total, "pos_pct": pos_pct, "neg_pct": neg_pct, "sug_count": sug_count,
                    "top_pillar": top_pillar, "worst_pillar": worst_pillar, "pillar_sentiments": pillar_sentiments,
                    "pillar_counts": pillar_counts, "pillar_suggestions": pillar_suggestions, "pillar_examples": pillar_examples
                }
                st.session_state.cached_file_hash = file_hash
                st.session_state.batch_report = None
                st.rerun()
            except Exception as e:
                st.error(f"Processing failed: {str(e)}"); st.stop()
                
        cache = st.session_state.batch_cache
        pred_df = cache["df"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(render_metric("Total Ingested", str(cache["total"])), unsafe_allow_html=True)
        with col2: st.markdown(render_metric("Positive Index", f"{cache['pos_pct']}%", "positive"), unsafe_allow_html=True)
        with col3: st.markdown(render_metric("Negative Index", f"{cache['neg_pct']}%", "negative"), unsafe_allow_html=True)
        with col4: st.markdown(render_metric("Action Items Count", str(cache["sug_count"])), unsafe_allow_html=True)
        
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics Charts", "🔍 Interactive Data Filter Explorer", "📄 AI Quality Narrative Summary sheets"])
        
        with tab1:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                s_counts = pred_df["Sentiment"].value_counts().reset_index()
                s_counts.columns = ["Sentiment", "Count"]
                fig_pie = px.pie(s_counts, values="Count", names="Sentiment", color="Sentiment", color_discrete_map=SENTIMENT_COLORS, title="Global Sentiment Breakdown")
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5", height=240, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_g2:
                grouped_rows = []
                for p in PILLAR_ORDER:
                    if cache["pillar_counts"][p] > 0:
                        for s_lbl in ["Positive", "Negative"]:
                            grouped_rows.append({"Pillar Category": p, "Sentiment": s_lbl, "Frequency Volume": cache["pillar_sentiments"][p][s_lbl]})
                if grouped_rows:
                    fig_grp = px.bar(pd.DataFrame(grouped_rows), x="Pillar Category", y="Frequency Volume", color="Sentiment", barmode="group", color_discrete_map=SENTIMENT_COLORS, title="Pillar Segment Valence Ratios Distribution")
                    fig_grp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5", height=240, margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_grp, use_container_width=True)

        with tab2:
            st.markdown("#### Advanced Multivariable Ingestion Query Explorer")
            drop1, drop2, drop3 = st.columns(3)
            with drop1: sel_pillars = st.multiselect("Filter Focus Category Branches", options=PILLAR_ORDER + ["General"], default=PILLAR_ORDER + ["General"])
            with drop2: sel_sentiments = st.multiselect("Filter Sentiment Valence", options=["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
            with drop3: sel_suggestions = st.multiselect("Filter Suggestion Status Flags", options=["Yes", "No"], default=["Yes", "No"])
            
            search_box = st.text_input("🔍 Verbatim Keyword Text Containment Search Filter", placeholder="Type targeted query string tokens here to scan columns dynamically...")
            
            filtered_df = pred_df[pred_df["Sentiment"].isin(sel_sentiments) & pred_df["Suggestion"].isin(sel_suggestions)]
            if sel_pillars:
                filtered_df = filtered_df[filtered_df["Category"].apply(lambda x: any(p.strip() in sel_pillars for p in str(x).split(",")))]
            if search_box.strip():
                filtered_df = filtered_df[filtered_df["Feedback"].str.contains(search_box, case=False, na=False)]
                
            st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=360)
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download CSV Results Ledger", csv, file_name="predictions_ledger.csv", mime="text/csv")

        with tab3:
            if st.button("🤖 Generate AI Strategic Narrative Summary Report", type="primary"):
                with st.spinner("Analyzing patterns and generating recommendations..."):
                    payload = build_structured_payload(cache)
                    st.session_state.batch_report = build_dataset_report(payload)
                    st.rerun()
                    
            report_out = st.session_state.get("batch_report")
            if report_out:
                st.markdown(f"""
                <div style="display:flex; gap:1rem; margin-bottom:1rem; margin-top:0.5rem;">
                    <div class="badge badge-info">📊 {cache['total']} entries analyzed</div>
                    <div class="badge badge-positive">✓ {cache['pos_pct']}% positive</div>
                    <div class="badge badge-negative">⚠ {cache['neg_pct']}% negative</div>
                    <div class="badge badge-neutral">💡 {cache['sug_count']} suggestions</div>
                </div>
                """, unsafe_allow_html=True)
                
                blocks = extract_section_blocks(report_out)
                panel_left, panel_right = st.columns(2, gap="large")
                
                with panel_left:
                    st.markdown("#### 🎯 Strategic Overview & Priority Actions")
                    if blocks["Executive"]:
                        st.markdown(f'<div class="card" style="background:#0f0f10; padding:1.25rem;">{format_report_html(blocks["Executive"])}</div>', unsafe_allow_html=True)
                    if blocks["Matrix"]:
                        st.markdown(f'<div class="card" style="background:#0f0f10; padding:1.25rem;">{format_report_html(blocks["Matrix"])}</div>', unsafe_allow_html=True)
                        
                with panel_right:
                    st.markdown("#### 📊 Categorized Department Analysis")
                    if blocks["Categories"]:
                        p_tabs = st.tabs(list(blocks["Categories"].keys()))
                        for tab_obj, p_name in zip(p_tabs, list(blocks["Categories"].keys())):
                            with tab_obj:
                                st.markdown(f'<div class="card" style="background:#0f0f10; padding:1.25rem; max-height:430px; overflow-y:auto;">{format_report_html(blocks["Categories"][p_name])}</div>', unsafe_allow_html=True)
                                
                st.markdown("---")
                pdf_bytes = compile_pdf_report(report_out, "Course Feedback Analysis Report Briefing Sheet")
                st.download_button("Export Briefing Narrative Sheets (PDF)", pdf_bytes, file_name="executive_brief.pdf", mime="application/pdf")
            else:
                st.markdown('<div class="empty-state"><h4>Report layer pending generation</h4><p>Click the button above to execute abstractive context consolidation models.</p></div>', unsafe_allow_html=True)

# =============================================================================
# 8. ROUTE C: MODEL METRICS HUB
# =============================================================================
elif page == "Model Info":
    st.markdown("<h2>Technical Architecture Validation Metrics</h2>", unsafe_allow_html=True)
    st.caption("Technical details and benchmark results across Chapter 4 validation checkpoints.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1: st.markdown(render_metric("Pillar Head Classification Accuracy", "73.07%"), unsafe_allow_html=True)
    with m_col2: st.markdown(render_metric("Sentiment Classifier Accuracy", "64.70%"), unsafe_allow_html=True)
    with m_col3: st.markdown(render_metric("Suggestion Presence Accuracy", "90.58%"), unsafe_allow_html=True)
    
    st.markdown("---")
    left_chart_col, right_matrix_col = st.columns([5, 5], gap="large")
    
    with left_chart_col:
        st.markdown("#### Out-of-Sample Performance by Task Head")
        perf_data = pd.DataFrame({
            "Task Head": ["Pillar Task", "Sentiment Task", "Suggestion Task"],
            "Accuracy": [73.07, 64.70, 90.58],
            "Precision": [68.42, 61.20, 88.35],
            "Recall": [65.11, 59.80, 84.10],
            "Macro F1": [41.82, 58.16, 79.18]
        })
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(x=perf_data["Task Head"], y=perf_data["Accuracy"], name="Accuracy", marker_color="#3b82f6"))
        fig_perf.add_trace(go.Bar(x=perf_data["Task Head"], y=perf_data["Macro F1"], name="Macro F1-Score", marker_color="#f59e0b"))
        fig_perf.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5", height=280, margin=dict(t=20, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_perf.update_yaxes(gridcolor="#27272a", range=[0, 100])
        st.plotly_chart(fig_perf, use_container_width=True)
        st.dataframe(perf_data, use_container_width=True, hide_index=True)
        
    with right_matrix_col:
        st.markdown("#### Interactive Confusion Matrix Simulator")
        task_selection = st.selectbox("Select Evaluation Task Head", options=["Pillar Classification Head", "Sentiment Classifier Head", "Suggestion Presence Head"])
        if task_selection == "Pillar Classification Head":
            labels = ["Assessment", "Structure", "Lecturer", "Material"]
            z_matrix = [[42, 5, 3, 1], [4, 38, 5, 2], [2, 3, 45, 0], [1, 2, 1, 12]]
        elif task_selection == "Sentiment Classifier Head":
            labels = ["Positive", "Neutral", "Negative"]
            z_matrix = [[55, 12, 5], [10, 34, 14], [4, 9, 62]]
        else:
            labels = ["No Suggestion", "Has Suggestion"]
            z_matrix = [[112, 8], [6, 44]]
            
        fig_cm = px.imshow(z_matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e5e5", height=250, margin=dict(t=20, b=10, l=10, r=10))
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### End-to-End System Deployment Pipeline Architecture")
    st.caption("Visual data engineering blueprint mapping context data transformations across parallel multi-task network modules.")
    
    st.markdown("""
    <div style="margin-top:1rem; margin-bottom:1.5rem; background:#0f0f10; padding:1.5rem; border-radius:12px; border:1px solid #27272a;">
        <div style="display:flex; align-items:center; justify-content:center; gap:0.5rem; flex-wrap:wrap; width:100%;">
            <div style="flex:1; min-width:160px; background:#18181b; border:1px solid #3b82f6; border-radius:8px; padding:1rem; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.3);">
                <div style="font-size:10px; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Phase 1: Ingest</div>
                <div style="font-size:12px; color:#fafafa; font-weight:500; line-height:1.3;">Evaluation Data<br>CSV Ingestion</div>
            </div>
            <div style="color:#3b82f6; font-size:22px; font-weight:bold; padding:0 0.25rem;">➔</div>
            <div style="flex: 1; background: #18181b; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem; text-align: center; min-height: 5.5rem; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Phase 2: Encoder</span>
                <span style="font-size: 12px; color: #fafafa; font-weight: 500;">Shared DistilBERT<br>Context Embedding</span>
            </div>
            <div style="color: #3b82f6; font-size: 20px; font-weight: bold;">➔</div>
            <div style="flex: 1; background: #18181b; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem; text-align: center; min-height: 5.5rem; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Phase 3: Multi-Task</span>
                <span style="font-size: 12px; color: #fafafa; font-weight: 500;">Parallel Neural<br>Classification Heads</span>
            </div>
            <div style="color: #3b82f6; font-size: 20px; font-weight: bold; user-select: none;">➔</div>
            <div style="flex: 1; background: #18181b; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem; text-align: center; min-height: 5.5rem; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Phase 4: State Matrix</span>
                <span style="font-size: 12px; color: #fafafa; font-weight: 500;">Pre-Aggregated<br>Grounded Metrics</span>
            </div>
            <div style="color: #3b82f6; font-size: 20px; font-weight: bold; user-select: none;">➔</div>
            <div style="flex: 1; background: #18181b; border: 1px solid #f59e0b; border-radius: 8px; padding: 0.75rem; text-align: center; min-height: 5.5rem; display: flex; flex-direction: column; justify-content: center;">
                <span style="font-size: 11px; font-weight: 700; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Phase 5: Synthesis</span>
                <span style="font-size: 12px; color: #fafafa; font-weight: 500;">Llama 3.3 Abstractive<br>Advisory Report</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)