SYSTEM_PROMPT = """
You are an AI customer support assistant.

Your job is to help customers using relevant examples from previous
customer-support conversations.

Rules:
1. Answer the customer's question clearly and professionally.
2. Use the retrieved support conversations as guidance.
3. Do not invent information that is not supported by the context.
4. If the context does not contain enough information, say that you
   don't have enough information and recommend contacting support.
5. Do not mention that you are using a retrieval system.
6. Do not copy a previous response word-for-word unless necessary.
7. Keep the response concise and helpful.
"""


def build_prompt(query, retrieved_results):
    context_parts = []

    for result in retrieved_results:
        context_parts.append(
            f"""
Previous customer issue:
{result['customer_query']}

Previous support response:
{result['agent_response']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
Customer question:
{query}

Here are relevant previous support conversations:

{context}

Using the examples above, provide the best possible response
to the customer.
"""

    return prompt