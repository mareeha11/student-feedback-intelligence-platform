# Student Feedback Intelligence Platform

An NLP-based application for analysing student feedback through aspect classification, sentiment analysis, summarisation, and recommendation generation.

## Overview

Student feedback contains valuable information about teaching quality, assessment practices, course structure, and learning materials. This project explores a multi-stage Natural Language Processing (NLP) pipeline that transforms feedback text into structured insights to support educational quality improvement.

## Key Features

- **Feedback classification:** Categorises feedback into Assessment, Course Structure, Lecturer, Subject Material, and Other.
- **Sentiment analysis:** Classifies feedback as positive, neutral, or negative.
- **Suggestion detection:** Identifies whether a feedback item contains a suggestion.
- **Topic modelling:** Uses BERTopic to explore recurring themes within feedback categories.
- **Summarisation and recommendations:** Generates concise summaries and actionable recommendations from selected feedback.
- **Interactive interface:** Built with Streamlit for single-feedback analysis, batch processing, and model evaluation.

## Technical Stack

Python · Streamlit · PyTorch · Transformers · DistilBERT · RoBERTa · SVM · BERTopic · Pandas · Plotly

## Model Evaluation

The multi-task DistilBERT model was evaluated on a held-out test set.

| Task | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Feedback category classification | 73.07% | 0.4182 |
| Sentiment classification | 64.70% | 0.5816 |
| Suggestion detection | 90.58% | 0.7918 |

Macro-F1 is included to reflect performance across classes rather than accuracy alone. Results are specific to the evaluation dataset and should be interpreted in the context of class imbalance and the dataset's characteristics.

## Project Structure

```text
feedback-deployment/
├── app.py
├── engine.py
├── groq_engine.py
├── summarizer_engine.py
├── test.py
├── requirements.txt
├── tokenizers/
│   ├── distilbert_tokenizer/
│   └── roberta_tokenizer/
└── dep/
    └── model/
        └── best_distilbert_model.pt  # stored locally; not tracked by Git
```

## Running Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Configure any required API credentials in a local `.env` file.
5. Ensure the trained model checkpoint is available at the path expected by the application.
6. Run the Streamlit application:

```bash
streamlit run app.py
```

Exact dependency versions, model setup, and API configuration instructions will be documented as the project is prepared for reproducible deployment.

## Model and Data Availability

The trained model checkpoint is not included in this repository because of its size. The research dataset is also not included. Access to any model or dataset will depend on applicable licensing, permission, and hosting arrangements.

## Project Status

The application currently runs locally. Public demo hosting and deployment configuration are planned next.

## Research Context

This project was developed as part of a Master's research project on student feedback analysis, summarisation, and recommendation generation.

## Author

**Mareeha Sultana Mohamed**

- GitHub: https://github.com/mareeha11
- LinkedIn: https://www.linkedin.com/in/mareeha-sultana