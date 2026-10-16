"""
Aggregation abstraction layer routing front-end Streamlit container variables 
to the Llama-3.3 LLM synthesis pipeline matrix.
"""
from groq_engine import (
    generate_single_feedback_report,
    generate_dataset_report
)

def build_single_report(
    ml_output,
    text
):
    """Abstraction function for single granular diagnostic report pipeline orchestration."""
    return generate_single_feedback_report(
        ml_output,
        text
    )

def build_dataset_report(
    prediction_text_summary
):
    """Abstraction function for corpora batch dataset synthesis report pipeline orchestration."""
    return generate_dataset_report(
        prediction_text_summary
    )