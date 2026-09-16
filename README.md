# SupportIQ — AI Customer Support Agent

> An evaluation-driven AI support agent that classifies customer requests, retrieves similar historical resolutions, generates grounded responses, and decides when a human should take over.

SupportIQ is built for the **AmazonHelp** brand using the Customer Support on Twitter dataset.

The system goes beyond a basic RAG chatbot by combining:

- Semantic intent classification
- Historical support-case retrieval
- Grounded response generation
- Risk-aware escalation
- Evaluation and human/LLM response review

---

## Architecture

```
Customer Message
       |
       v
Intent Classification
       |
       v
Risk Detection
       |
       v
BGE + FAISS Retrieval
       |
       v
Confidence + Decision
      / \
     /   \
AI Handle  Human Escalation
    |
    v
Grounded LLM Response
```

---

## Key Features

### Intent Classification

Nine AmazonHelp-specific intents:

- `delivery_tracking`
- `order_management`
- `returns_refunds`
- `payment_billing`
- `product_information`
- `technical_support`
- `account_subscription`
- `seller_marketplace`
- `other`

The production prototype uses a BGE semantic-centroid classifier.

### Historical Resolution Retrieval

The system retrieves similar historical customer/agent interactions from a corpus of:

**148,556 customer-query / agent-response pairs**

using:

- `BAAI/bge-small-en-v1.5`
- 384-dimensional embeddings
- FAISS `IndexFlatIP`

### Grounded Generation

The LLM uses retrieved historical resolutions as context and is instructed not to invent:

- Refund policies
- Prices
- Delivery dates
- Account information
- Actions it did not perform

### Human Escalation

The system can escalate:

- Financial issues
- Account-security issues
- Security concerns
- High-risk cases
- Ambiguous requests
- Low-confidence cases

The UI displays the decision and reason.

---

## Evaluation

A 200-example hand-labelled golden set was created from AmazonHelp conversations.

### Intent Classification

Four-fold stratified cross-validation:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority Class | 31.00% | 5.26% |
| TF-IDF + Logistic Regression | 44.50% | 20.56% |
| TF-IDF + Linear SVM | 44.50% | 19.82% |
| BGE Semantic Centroid | 53.00% | 40.63% |
| BGE k-NN (k=3) | 46.50% | 32.01% |

The BGE semantic-centroid model was selected for the production prototype.

### Retrieval

| Metric | Result |
| --- | ---: |
| Recall@5 | 89.70% |
| Recall@10 | 93.33% |
| MRR@5 | 86.77% |
| MRR@10 | 87.27% |

165 of the 200 golden queries were evaluable for this benchmark.

### Escalation

| Metric | Result |
| --- | ---: |
| Accuracy | 85.00% |
| Precision | 92.86% |
| Recall | 86.67% |
| F1 | 89.66% |

Only 1 false positive and 2 false negatives occurred in the safety benchmark.

### Response Quality

Human evaluation of the available real Gemini response:

**3.67/5 overall**

LLM-as-judge:

**4.67/5 overall**

The LLM-judge result is currently n=1, so it is reported as a pilot result rather than a general benchmark.

### Important Limitation

The 89.70% Recall@5 result does not mean 89.7% of final answers are correct.

It measures whether a relevant historical resolution appears in the top five for evaluable retrieval queries.

Similarly, the 4.67/5 LLM-judge result is based on only one real generated response.

Detailed methodology, failure analysis and the one-week improvement plan are available in:

`evaluation/HIVER_REPORT.md`

---

## Quick Start

### Backend

```bash
git clone <YOUR_GITHUB_REPOSITORY>
cd supportIQ

python -m venv .venv
source .venv/Scripts/activate

pip install -r requirements.txt

uvicorn api.main:app --reload
```

For Windows CMD:

```cmd
.venv\Scripts\activate
```

Create `.env`:

```
MOCK_LLM=true
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL shown by Next.js.

---

## Evaluation Commands

```bash
# Intent baselines
python -m evaluation.evaluate_intent_classifier

# BGE semantic intent
python -m evaluation.evaluate_semantic_intent

# BGE k-NN comparison
python -m evaluation.evaluate_semantic_knn

# Retrieval + escalation
python -m evaluation.evaluate_escalation

# Response quality
python -m evaluation.evaluate_response_quality

# LLM-as-judge
python -m evaluation.judge_responses
```

---

## Project Structure

```
supportIQ/
├── api/                    # FastAPI backend
├── src/                    # Retrieval, RAG, intent & escalation
├── evaluation/             # Evaluation scripts and results
├── data/                   # Dataset and retrieval artifacts
├── notebooks/              # Data exploration and corpus building
├── frontend/               # Next.js support console
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

- **Backend:** Python, FastAPI, Pandas, NumPy
- **AI:** BGE, Sentence Transformers, RAG, Gemini-compatible LLM
- **Retrieval:** FAISS, vector similarity search
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Evaluation:** scikit-learn, cross-validation, Recall@K, MRR, human evaluation, LLM-as-judge

---

## Design Philosophy

A support agent should know not only how to answer, but also when it should not answer.

SupportIQ therefore evaluates retrieval, intent classification, generation and escalation as separate components rather than relying on a single LLM quality score.

---

## Detailed Evaluation Report

For the complete technical evaluation, methodology, failure analysis, limitations, and one-week improvement plan, see:

**[Hiver Take-Home Evaluation Report](evaluation/HIVER_REPORT.md)**

---

## Author

**Ritesh Kushwaha**

B.Tech — Computer Science Engineering