import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)

try:
    client = get_client()
except Exception:
    client = None

def generate_single_feedback_report(ml_output: dict, text: str) -> str:
    """Generate a split summary and recommendation brief for a single feedback entry."""
    if not client:
        return "Groq client unavailable."
    pillars = ml_output.get("pillars", [])
    sentiment = ml_output.get("sentiment", "Neutral")
    suggestion = ml_output.get("suggestion", "No")
    pillar_str = ", ".join(pillars) if pillars else "General"
    
    prompt = f"""You are a university quality assurance specialist reviewing a single line of student feedback.
FEEDBACK: "{text}"
CLASSIFICATION:
- Category: {pillar_str}
- Sentiment: {sentiment}
- Contains explicit suggestion: {suggestion}

Generate an operational diagnostic brief using this exact format. Limit each line to a maximum of 25 words:

Summary: (State the core friction point or praise directly)
Recommendation: (Provide one concrete, actionable pedagogical step the tutor can take)"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"

def generate_dataset_report(structured_payload: dict) -> str:
    """Generate a comprehensive tutor-facing report from pre-aggregated feedback data."""
    if not client:
        return "Groq client unavailable."
        
    total = structured_payload.get("total_records", 0)
    pos_pct = structured_payload.get("positive_pct", 0)
    neg_pct = structured_payload.get("negative_pct", 0)
    sug_count = structured_payload.get("suggestion_count", 0)
    top_pillar = structured_payload.get("top_pillar", "N/A")
    risk_pillar = structured_payload.get("risk_pillar", "N/A")
    pillar_data = structured_payload.get("pillar_data", {})
    
    pillar_sections = []
    for pillar_name, data in pillar_data.items():
        pos_list = data.get('positive_examples', [])[:3]
        neg_list = data.get('negative_examples', [])[:3]
        sug_list = data.get('suggestion_examples', [])[:3]
        
        pos_str = "\n".join(f"- {fb}" for fb in pos_list) if pos_list else "- None recorded"
        neg_str = "\n".join(f"- {fb}" for fb in neg_list) if neg_list else "- None recorded"
        sug_str = "\n".join(f"- {fb}" for fb in sug_list) if sug_list else "- None recorded"
        
        section = f"""
### {pillar_name}
- Total entries: {data.get('count', 0)}
- Sentiment breakdown: {data.get('positive', 0)} positive, {data.get('neutral', 0)} neutral, {data.get('negative', 0)} negative

Sample positive feedback strings:
{pos_str}

Sample negative feedback strings:
{neg_str}

Student suggestions strings:
{sug_str}
"""
        pillar_sections.append(section)
        
    pillar_context = "\n".join(pillar_sections)
    
    prompt = f"""You are a senior educational quality consultant preparing a decision-support report for university tutors and course coordinators.
## DATASET OVERVIEW
- Total feedback entries analyzed: {total}
- Overall sentiment: {pos_pct}% positive, {neg_pct}% negative
- Entries containing actionable suggestions: {sug_count}
- Most discussed topic: {top_pillar}
- Highest-risk area (highest negative ratio): {risk_pillar}

## CATEGORIZED FEEDBACK DATA
{pillar_context}
---
Generate a professional tutor-facing report with this exact structure:

# EXECUTIVE SUMMARY
Write a concise 3-sentence evaluation synthesis. Sentence 1 defines overall course health based on the strings. Sentence 2 isolates the primary operational strength to protect. Sentence 3 declares the most urgent student challenge requiring immediate programmatic triage. Avoid generic filler.

# CATEGORY ANALYSIS
For each category that has data, provide:

## [Category Name]
**Current State**: A concise 2-sentence breakdown detailing exactly what students are experiencing in this area, based on the feedback patterns.
**Strengths to Maintain**:
- [Specific positive behavior or resource students valued]
**Issues Requiring Attention**:
- [Concrete problem derived from negative feedback]
**Recommended Actions**:
- IMMEDIATE (next 2 weeks): [Specific tactical fix]
- SHORT-TERM (this semester): [Structural adjustment]
- LONG-TERM (curriculum review): [Strategic improvement]

# PRIORITY ACTION MATRIX
Create a ranked list of the 3-5 most impactful, specific actions across all categories, ordered by urgency. 
CRITICAL FORMANTING LAW: You MUST follow the delimiter template format below for every single item. Do not truncate or leave out fields.

Format each numbered item exactly like this example:
1. [Write the concrete action step here] — [Write single category here] — [Write expected impact here]

---
CRITICAL COMPILER INSTRUCTIONS:
- NEVER comment on data constraints, entry numbers, logging ratios, or missing examples.
- Base every single observation on the text strings provided. Do not use generic filler.
- Ensure that the category and impact elements are printed explicitly on the same line as the action step, separated by em-dashes ( — )."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation failed: {str(e)}"