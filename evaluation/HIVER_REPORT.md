# SupportIQ

### Hiver SDE Intern Take-Home Report

---

## Table of Contents

- [SupportIQ](#supportiq)
  - [Hiver SDE Intern Take-Home Report](#hiver-sde-intern-take-home-report)
- [Table of Contents](#table-of-contents)
- [1. Problem Framing](#1-problem-framing)
- [2. Dataset & Golden Evaluation Set](#2-dataset--golden-evaluation-set)
- [3. System Architecture](#3-system-architecture)
- [4. Baselines & Model Selection](#4-baselines--model-selection)
- [5. Evaluation Results](#5-evaluation-results)
  - [Retrieval](#retrieval)
  - [Escalation](#escalation)
  - [Response quality](#response-quality)
- [6. Top 5 Failure Modes](#6-top-5-failure-modes)
  - [1. Promotional/non-question messages](#1-promotionalnon-question-messages)
  - [2. Social-media retrieval noise](#2-social-media-retrieval-noise)
  - [3. Multilingual weakness](#3-multilingual-weakness)
  - [4. Account-security escalation](#4-account-security-escalation)
  - [5. Retrieval evaluation coverage](#5-retrieval-evaluation-coverage)
- [7. What Is Misleading About My Headline Number?](#7-what-is-misleading-about-my-headline-number)
- [8. What I Would Build With One More Week](#8-what-i-would-build-with-one-more-week)
- [9. Decision Log](#9-decision-log)
- [Conclusion](#conclusion)

---

## 1. Problem Framing

Customer-support automation has two failure modes that matter more than simply producing fluent text: retrieving the wrong historical resolution, and confidently answering cases that should go to a human.

SupportIQ is an evaluation-driven support agent for **AmazonHelp** using the Customer Support on Twitter dataset. For each incoming message it:

- classifies the support intent;
- retrieves similar historical customer/agent interactions;
- generates a grounded response when appropriate;
- detects financial, security, high-risk and ambiguous cases;
- combines retrieval confidence with risk rules to decide AI handling vs human escalation;
- exposes evidence and decision rationale in the UI.

The design goal is **safe automation with measurable evidence**, not maximum automation.

---

## 2. Dataset & Golden Evaluation Set

AmazonHelp contains 169,840 brand tweets, 82,534 conversations and 374,042 tweets in those conversations. Mean conversation length is 4.53 tweets and median is 3.

A **200-example hand-labelled golden set** was created for evaluation. It contains 17 Japanese examples and 183 non-Japanese examples.

The frozen nine-intent taxonomy is:

1. `delivery_tracking`
2. `order_management`
3. `returns_refunds`
4. `payment_billing`
5. `product_information`
6. `technical_support`
7. `account_subscription`
8. `seller_marketplace`
9. `other`

The retrieval corpus contains **148,556 customer-query / historical-agent-response pairs**. Production retrieval uses `BAAI/bge-small-en-v1.5`, 384-dimensional normalized embeddings and FAISS `IndexFlatIP`.

---

## 3. System Architecture

```
Customer Message
       |
       v
Intent Classification (BGE semantic model)
       |
       v
Risk Detection
       |
       v
Semantic Retrieval (BGE + FAISS)
       |
       v
Confidence + Decision
      /  AI handle  Human escalation
     |             |
 Grounded LLM   Reason + risk
     |
     v
Agent Console
```

The UI shows retrieved historical sources, semantic score, risk categories and the AI-vs-human decision.

---

## 4. Baselines & Model Selection

Four-fold stratified cross-validation was used on the 200 labelled examples.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Majority class | 31.00% | 5.26% | 14.67% |
| TF-IDF + Logistic Regression | 44.50% | 20.56% | 37.84% |
| TF-IDF + Linear SVM | 44.50% | 19.82% | 36.99% |
| **BGE semantic centroid** | **53.00%** | **40.63%** | **52.07%** |
| BGE k-NN, best k=3 | 46.50% | 32.01% | 43.94% |

BGE semantic centroid was selected because it produced the best accuracy and Macro-F1. Its Macro-F1 is almost twice the Logistic Regression baseline.

---

## 5. Evaluation Results

### Retrieval

Of 200 golden queries, 165 were evaluable because 35 expected response IDs were not present in the constructed retrieval corpus.

| Metric | Result |
| --- | ---: |
| Recall@5 | **89.70%** |
| Recall@10 | **93.33%** |
| MRR@5 | **86.77%** |
| MRR@10 | **87.27%** |

### Escalation

| Metric | Result |
| --- | ---: |
| Accuracy | **85.00%** |
| Precision | **92.86%** |
| Recall | **86.67%** |
| F1 | **89.66%** |
| False positives | **1** |
| False negatives | **2** |

Category accuracy: Financial **100%**, Ambiguous **100%**, Security **80%**, Normal **80%**, Account security **66.7%**.

Across 200 golden queries, the mean retrieval-derived semantic score was **0.8903** (min 0.7748, max 0.9546). This is a similarity-derived signal, **not a calibrated probability**.

### Response quality

Human review of the available real Gemini response: Relevance **5/5**, Groundedness **4/5**, Helpfulness **2/5**, Overall **3.67/5**.

LLM judge on the same response: Relevance **5/5**, Groundedness **5/5**, Helpfulness **4/5**, Overall **4.67/5**.

The LLM-judge result is **n=1**, so it is a pilot observation, not a general benchmark. The difference between human and judge scores also shows why judge-human agreement needs a larger jointly labelled sample.

---

## 6. Top 5 Failure Modes

### 1. Promotional/non-question messages

**Example:** "Watch amazon and Nokia #UniteFor #Fun by bringing the sensational new Nokia 6 to fans all across India."

Retrieval found relevant Nokia 6 examples, but the response was generic rather than recognizing that this was primarily promotional/non-actionable content.

**Hypothesis:** topical retrieval relevance does not mean a message requires customer support.

**Fix:** add a support-request vs non-request gate before response generation.

### 2. Social-media retrieval noise

**Example:** "Wow Dept.: GMgr kept giving me 20 hours/week when I repeatedly asked for 40..."

The corpus contains social posts where semantic similarity can retrieve topical but operationally irrelevant content.

**Fix:** metadata-aware filtering, conversation-role filtering and a second-stage reranker.

### 3. Multilingual weakness

**Example:** "Hab mir Star Wars Bettlefront 2 vorbestellt und es ist immernoch nicht angekommen".

The golden set contains 17 Japanese examples as well as German/French examples. The production embedding model is English-focused.

**Fix:** benchmark an explicitly multilingual embedding model and report language-stratified results.

### 4. Account-security escalation

Account-security accuracy was only **66.7%**, lower than financial and ambiguous categories.

**Hypothesis:** indirect security wording overlaps with normal account/order language.

**Fix:** dedicated security classifier plus adversarial examples for hacked accounts, changed passwords, unauthorized access and suspicious activity.

### 5. Retrieval evaluation coverage

35 of 200 golden examples were not evaluable for exact expected-response retrieval.

**Fix:** preserve the full conversation graph and construct deterministic mappings from every golden customer tweet to its actual subsequent agent response when available.

---

## 7. What Is Misleading About My Headline Number?

The most tempting headline is **89.70% Recall@5**. It is useful, but it is not the probability that SupportIQ will produce a correct answer.

Only 165/200 examples were evaluable; Recall@5 measures whether a relevant historical resolution appears in the top five, not whether the generated answer is correct; and real-LLM response-quality evaluation was limited by API quota.

Likewise, **4.67/5 LLM-judge quality is n=1**.

Therefore the accurate interpretation is:

> SupportIQ retrieves a relevant historical resolution in the top five for 89.7% of evaluable golden queries, but retrieval quality does not by itself guarantee generation quality or safe automation.

---

## 8. What I Would Build With One More Week

1. **Intent:** larger balanced annotation set and hard negatives for closely related intents.
2. **Retrieval:** BGE candidate retrieval followed by a cross-encoder reranker.
3. **Request gate:** distinguish support requests from promotional/social/non-actionable messages.
4. **Escalation:** dedicated security classifier and calibrated threshold policy optimized for high-risk recall.
5. **Evaluation:** 50–100 real-LLM responses jointly scored by humans and the judge; measure correlation and within-one-point agreement.
6. **Reproducibility:** deterministic preparation, pinned dependencies and one-command evaluation.

---

## 9. Decision Log

1. **Chose AmazonHelp as the brand**
   - Kept the project focused on one brand as required.
   - This allowed the intent taxonomy and retrieval corpus to be domain-specific.

2. **Used a 9-intent taxonomy**
   - Kept the taxonomy small enough to have meaningful examples per class.
   - Sparse categories were merged into broader operational intents.

3. **Used BGE-small for semantic retrieval**
   - It is a practical 384-dimensional embedding model that runs locally on CPU and supports both retrieval and semantic intent classification.

4. **Used FAISS IndexFlatIP**
   - Embeddings are normalized, making inner product equivalent to cosine similarity.
   - This provides simple and deterministic retrieval for the corpus size.

5. **Used historical resolutions for response grounding**
   - Similar historical conversations provide concrete evidence for how related cases were handled.

6. **Added an explicit escalation layer**
   - Financial, account-security, high-risk and ambiguous cases should not rely solely on generated responses.

7. **Used semantic similarity as an escalation signal**
   - Low similarity can indicate that the system lacks a sufficiently similar historical case.
   - This signal is treated as a semantic score, not a calibrated probability.

8. **Did not automatically adopt the 0.90 confidence threshold**
   - Although it improved precision in threshold calibration, it would send a substantially larger fraction of cases to humans.
   - I treated threshold selection as a policy trade-off rather than automatically choosing the highest-precision threshold.

9. **Used a 200-example golden set**
   - This is within the required 150–250 range while keeping manual labeling manageable.

10. **Used stratified 4-fold cross-validation**
    - This reduces dependence on a single train/test split for the relatively small labeled dataset.

11. **Compared trivial and simple baselines**
    - Majority-class and TF-IDF models provide reference points for measuring whether semantic embeddings add value.

12. **Used human review alongside LLM judging**
    - Human scoring provides a reference for checking the behavior of the automated judge.
    - The current jointly scored sample is only n=1, so it is treated as a pilot rather than a general agreement benchmark.

13. **Constrained generated responses to retrieved evidence**
    - The system is instructed not to invent refunds, prices, delivery dates, account information, or actions it did not perform.

14. **Documented multilingual limitations**
    - The production embedding model is English-focused, so multilingual messages remain a known limitation rather than being hidden from evaluation.

15. **Separated the README from the detailed evaluation**
    - The README is optimized for quick review, while this report contains the detailed methodology, analysis, failure modes and limitations.

### Engineering Limitations

- The golden set contains 200 examples, which is useful for a take-home evaluation but still small for broad generalization.
- 35 golden examples were not evaluable for exact-response retrieval because their expected response IDs were not present in the constructed retrieval corpus.
- The production embedding model is English-focused.
- Semantic similarity scores are not calibrated probabilities.
- Real-LLM response-quality evaluation was limited by API quota.
- The current judge-human comparison is n=1 and is therefore insufficient to establish reliable judge-human agreement.

One methodological distinction is important: the reported BGE intent metrics use four-fold cross-validation, while the production prototype fits intent centroids on the full labelled set for the demo. These should not be conflated.

---

## Conclusion

SupportIQ demonstrates an end-to-end support automation system where **retrieval, intent, generation and escalation are evaluated separately**.

The strongest result is not one number. It is the combination of strong top-k retrieval, measurable improvement over simple intent baselines, high-precision escalation, grounded generation, visible evidence and explicit failure analysis.

The next stage is multilingual support, retrieval reranking, larger human/LLM evaluation and calibrated automation policy.
