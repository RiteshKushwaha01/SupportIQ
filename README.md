# SupportIQ --- AI Customer Support Agent

> An evaluation-driven AI support agent that classifies customer
> requests, retrieves similar historical resolutions, generates grounded
> responses, and decides when a human should take over.

SupportIQ is built for the **AmazonHelp** brand using the Customer
Support on Twitter dataset.

The system goes beyond a basic RAG chatbot by combining:

-   Semantic intent classification
-   Historical support-case retrieval
-   Grounded response generation
-   Risk-aware escalation
-   Evaluation and human/LLM response review

------------------------------------------------------------------------

## Project Goals

SupportIQ is designed around four concrete support-automation tasks:

1.  **Classification** --- identify the customer's support intent.
2.  **Retrieval** --- find similar historical customer/agent
    resolutions.
3.  **Generation** --- draft a response grounded in those historical
    cases.
4.  **Escalation** --- route risky, ambiguous or low-confidence requests
    to a human.

The system therefore separates *finding evidence* from *deciding whether
to answer* and from *generating the answer*.

------------------------------------------------------------------------

## Architecture

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

------------------------------------------------------------------------

## Key Features

### Intent Classification

Nine AmazonHelp-specific intents:

-   `delivery_tracking`
-   `order_management`
-   `returns_refunds`
-   `payment_billing`
-   `product_information`
-   `technical_support`
-   `account_subscription`
-   `seller_marketplace`
-   `other`

The production prototype uses a BGE semantic-centroid classifier.

### Historical Resolution Retrieval

The system retrieves similar historical customer/agent interactions from
a corpus of:

**148,556 customer-query / agent-response pairs**

using:

-   `BAAI/bge-small-en-v1.5`
-   384-dimensional embeddings
-   FAISS `IndexFlatIP`

### Grounded Generation

The LLM uses retrieved historical resolutions as context and is
instructed not to invent:

-   Refund policies
-   Prices
-   Delivery dates
-   Account information
-   Actions it did not perform

### Human Escalation

The system can escalate:

-   Financial issues
-   Account-security issues
-   Security concerns
-   High-risk cases
-   Ambiguous requests
-   Low-confidence cases

The UI displays the decision and reason.

------------------------------------------------------------------------

## Evaluation

A 200-example hand-labelled golden set was created from AmazonHelp
conversations.

### Intent Classification

Four-fold stratified cross-validation:

  Model                            Accuracy   Macro-F1
  ------------------------------ ---------- ----------
  Majority Class                     31.00%      5.26%
  TF-IDF + Logistic Regression       44.50%     20.56%
  TF-IDF + Linear SVM                44.50%     19.82%
  BGE Semantic Centroid              53.00%     40.63%
  BGE k-NN (k=3)                     46.50%     32.01%

The BGE semantic-centroid model was selected for the production
prototype.

### Retrieval

  Metric        Result
  ----------- --------
  Recall@5      89.70%
  Recall@10     93.33%
  MRR@5         86.77%
  MRR@10        87.27%

165 of the 200 golden queries were evaluable for this benchmark.

### Escalation

  Metric        Result
  ----------- --------
  Accuracy      85.00%
  Precision     92.86%
  Recall        86.67%
  F1            89.66%

Only 1 false positive and 2 false negatives occurred in the safety
benchmark.

### Response Quality

Human evaluation of the available real Gemini response:

**3.67/5 overall**

LLM-as-judge:

**4.67/5 overall**

The LLM-judge result is currently n=1, so it is reported as a pilot
result rather than a general benchmark.

### Important Limitation

The 89.70% Recall@5 result does not mean 89.7% of final answers are
correct.

It measures whether a relevant historical resolution appears in the top
five for evaluable retrieval queries.

Similarly, the 4.67/5 LLM-judge result is based on only one real
generated response.

Detailed methodology, failure analysis and the one-week improvement plan
are available in:

`evaluation/HIVER_REPORT.md`

------------------------------------------------------------------------

## Quick Start

The recommended local setup is **Docker for the backend + Next.js for
the frontend**. This reproduces the deployment runtime and automatically
downloads the large retrieval artifacts from Hugging Face.

### Prerequisites

Install:

-   Git
-   Docker Desktop
-   Node.js / npm

### 1. Clone the repository

``` bash
git clone https://github.com/RiteshKushwaha01/SupportIQ.git
cd SupportIQ
```

### 2. Configure the backend

Create `.env` in the project root:

``` env
GEMINI_API_KEY=
MOCK_LLM=true

FRONTEND_URLS=http://localhost:3000
FRONTEND_ORIGIN_REGEX=https://.*\\.vercel\\.app
WARMUP_ON_START=true

HF_ARTIFACTS_REPO=RiteshKushwaha01/supportiq-artifacts
HF_ARTIFACTS_REPO_TYPE=model
```

`MOCK_LLM=true` lets you test retrieval, intent classification and
escalation without consuming Gemini quota.

For real generated responses:

``` env
MOCK_LLM=false
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env` or the API key.

### 3. Build the backend image

``` bash
docker build -t supportiq-api .
```

### 4. Start the backend

Git Bash:

``` bash
docker run --rm -p 7860:7860 \
  --env-file .env \
  -e FRONTEND_URLS=http://localhost:3000 \
  -e WARMUP_ON_START=true \
  supportiq-api
```

Windows CMD:

``` cmd
docker run --rm -p 7860:7860 ^
  --env-file .env ^
  -e FRONTEND_URLS=http://localhost:3000 ^
  -e WARMUP_ON_START=true ^
  supportiq-api
```

On first startup the backend downloads `retrieval.index` and
`retrieval_corpus.db`, loads FastEmbed, initializes FAISS and loads the
intent classifier.

Wait for:

``` text
RAG pipeline ready!
LIFESPAN: Warmup complete.
Application startup complete.
```

### 5. Verify the backend

Open:

``` text
http://localhost:7860/api/health
```

or:

``` bash
curl http://localhost:7860/api/health
```

Expected:

``` json
{
  "status": "healthy",
  "service": "supportiq-api",
  "ready": true,
  "warmup_on_start": true
}
```

### 6. Start the frontend

Open another terminal:

``` bash
cd frontend
npm install
```

Create:

``` text
frontend/.env.local
```

with:

``` env
NEXT_PUBLIC_API_URL=http://localhost:7860
```

Then:

``` bash
npm run dev
```

Open:

``` text
http://localhost:3000
```

### 7. Test the API directly

``` bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is my package?","top_k":5}'
```

------------------------------------------------------------------------

## Local development without Docker

Docker is recommended, but the backend can also be run directly with
Python.

``` bash
python -m venv .venv
```

Windows Git Bash:

``` bash
source .venv/Scripts/activate
```

Windows CMD:

``` cmd
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Create `.env` using the configuration above, then:

``` bash
uvicorn api.main:app --reload --port 7860
```

The API will be available at:

``` text
http://localhost:7860
```

The first startup still needs the Hugging Face artifacts and FastEmbed
model.

------------------------------------------------------------------------

## Deployment Architecture

The backend is intentionally separated from the frontend and large
generated artifacts:

``` text
                    Vercel
              Next.js / React UI
                      |
                      | HTTPS
                      v
          Linux VM / Docker Host
          ----------------------
          FastAPI
          FastEmbed
          FAISS
          SQLite
          Gemini client
                      |
                      | startup download
                      v
              Hugging Face Hub
              ----------------
              retrieval.index
              retrieval_corpus.db
```

### Why this backend is deployable

The complete runtime needs substantially more memory than tiny 256--512
MB serverless/free PaaS instances. Instead of forcing the whole project
into a small function/container, SupportIQ uses a normal Docker process
on a VM/container host with sufficient RAM.

Several changes make deployment practical:

1.  **Large artifacts are externalized.**\
    The FAISS index and SQLite retrieval corpus are stored on Hugging
    Face rather than committed to Git.

2.  **SQLite replaces full-corpus runtime loading.**\
    FAISS returns row IDs and SQLite fetches only the matching
    historical records.

3.  **FastEmbed is used for embedding inference.**\
    The deployed runtime uses the ONNX-based FastEmbed path instead of
    the heavier PyTorch/Sentence-Transformers runtime.

4.  **Docker provides reproducibility.**\
    The same Docker image can run locally or on a Linux VM/container
    host.

5.  **FastAPI warmup makes readiness explicit.**\
    Startup initializes artifacts, the embedding model, FAISS and the
    intent model before the service reports itself ready.

### Backend deployment

A suitable VM/container host should provide enough RAM for the FastAPI
process, FastEmbed model, FAISS index and Python runtime.

On a Linux server:

``` bash
git clone https://github.com/RiteshKushwaha01/SupportIQ.git
cd SupportIQ

docker build -t supportiq-api .

docker run -d \
  --name supportiq-api \
  --restart unless-stopped \
  -p 8000:7860 \
  --env-file .env \
  supportiq-api
```

Check:

``` bash
docker logs -f supportiq-api
```

Then:

``` text
http://SERVER_IP:8000/api/health
```

For a public deployment, use HTTPS/reverse proxy or a secure tunnel
rather than exposing plain HTTP directly.

### Vercel configuration

Set:

``` env
NEXT_PUBLIC_API_URL=https://YOUR_BACKEND_DOMAIN
```

Do not add a trailing slash.

The backend should allow the Vercel origin through `FRONTEND_URLS` or
the configured origin regex.

------------------------------------------------------------------------

## Hugging Face Runtime Artifacts

The large deployment files are stored in:

``` text
RiteshKushwaha01/supportiq-artifacts
```

The backend downloads:

``` text
retrieval.index
retrieval_corpus.db
```

through:

``` text
scripts/download_artifacts.py
```

The public artifact repository means a fresh deployment does not need an
HF token just to download the public files. An unauthenticated-request
warning is not itself a deployment failure.

------------------------------------------------------------------------

## API Endpoints

### Health

``` http
GET /api/health
```

Reports whether the application is running and whether the RAG pipeline
has finished warming up.

### Chat

``` http
POST /api/chat
```

Example request:

``` json
{
  "query": "Where is my package?",
  "top_k": 5
}
```

The response contains the generated answer plus intent, confidence,
escalation, risk and retrieved-source information.

------------------------------------------------------------------------

## Troubleshooting

### `Failed to fetch`

Check:

``` text
http://localhost:7860/api/health
```

Then verify:

``` text
frontend/.env.local
```

contains:

``` env
NEXT_PUBLIC_API_URL=http://localhost:7860
```

Restart Next.js after changing `.env.local`.

### Port 7860 already allocated

Run:

``` bash
docker ps
```

If a SupportIQ container is already using port 7860, do not start
another one. Stop the existing container only if you need to restart it:

``` bash
docker stop <container_name>
```

### Gemini quota exceeded

A Gemini quota error means generation is unavailable for the configured
API project; it does not mean FAISS retrieval or FastAPI is broken.

For development:

``` env
MOCK_LLM=true
```

For real generation:

``` env
MOCK_LLM=false
GEMINI_API_KEY=your_key
```

### Backend still warming up

Run:

``` bash
docker logs -f supportiq-api
```

Wait for:

``` text
RAG pipeline ready!
LIFESPAN: Warmup complete.
Application startup complete.
```

### Hugging Face warning

If the artifact repository is public, unauthenticated downloads are
supported. The warning about rate limits is informational unless the
download itself fails.

------------------------------------------------------------------------

## Evaluation Commands

``` bash
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

------------------------------------------------------------------------

## Project Structure

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

------------------------------------------------------------------------

## Tech Stack

-   **Backend:** Python, FastAPI, Pandas, NumPy
-   **AI:** FastEmbed, BGE-small, RAG, Gemini-compatible LLM
-   **Retrieval:** FAISS, vector similarity search
-   **Frontend:** Next.js, React, TypeScript, Tailwind CSS
-   **Evaluation:** scikit-learn, cross-validation, Recall@K, MRR, human
    evaluation, LLM-as-judge

------------------------------------------------------------------------

## Design Philosophy

A support agent should know not only how to answer, but also when it
should not answer.

SupportIQ therefore evaluates retrieval, intent classification,
generation and escalation as separate components rather than relying on
a single LLM quality score.

------------------------------------------------------------------------

## Detailed Evaluation Report

For the complete technical evaluation, methodology, failure analysis,
limitations, and one-week improvement plan, see:

**[Hiver Take-Home Evaluation Report](evaluation/HIVER_REPORT.md)**

------------------------------------------------------------------------

## Author

**Ritesh Kushwaha**

B.Tech --- Computer Science Engineering
