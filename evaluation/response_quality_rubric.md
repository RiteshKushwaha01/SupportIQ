# SupportIQ Response Quality Rubric

## Purpose

Evaluate the quality of generated customer-support responses
using three dimensions:

1. Relevance
2. Groundedness
3. Helpfulness

Each dimension is scored from 1 to 5.

---

## 1. Relevance

Measures whether the response directly addresses the customer's
question or problem.

### Score 5
Directly answers the customer's question and stays focused.

### Score 4
Answers the question with minor unnecessary information.

### Score 3
Partially addresses the question but misses an important aspect.

### Score 2
Mostly unrelated or only weakly addresses the question.

### Score 1
Does not address the customer's question.

---

## 2. Groundedness

Measures whether factual claims in the response are supported
by the retrieved support context.

### Score 5
All important claims are clearly supported by retrieved context.

### Score 4
Almost completely supported; minor unsupported wording.

### Score 3
Some claims are supported but others are uncertain.

### Score 2
Contains significant unsupported claims.

### Score 1
Contains major fabricated or contradictory information.

---

## 3. Helpfulness

Measures whether the response gives the customer a useful
next step or resolution.

### Score 5
Clear, actionable, professional, and likely to resolve the issue.

### Score 4
Helpful and actionable but could be slightly clearer.

### Score 3
Provides some useful information but lacks a strong next step.

### Score 2
Limited practical value.

### Score 1
Not useful or likely to frustrate the customer.

---

## Overall Score

Overall quality is calculated as:

overall_score =
    (relevance + groundedness + helpfulness) / 3

Maximum score: 5.0

---

## Evaluation Principles

- Do not reward fluent language alone.
- Penalize unsupported factual claims.
- Penalize claims that the assistant performed an action
  when it did not actually perform that action.
- Prefer concise and actionable answers.
- A response should not invent refunds, delivery dates,
  account changes, prices, policies, or other unsupported facts.
- Human-escalated cases should not be judged as failed
  AI responses because no AI answer was generated.