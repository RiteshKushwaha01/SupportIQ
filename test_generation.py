from src.rag_pipeline import RAGPipeline


print("Starting RAG + LLM test...")

pipeline = RAGPipeline()

query = "My package has not arrived yet. Can you help me?"

result = pipeline.generate_response(query)

print("\n" + "=" * 60)
print("CUSTOMER:")
print(result["query"])

print("\nAI RESPONSE:")
print(result["answer"])

print("\nRETRIEVED EXAMPLES:")
for i, item in enumerate(result["retrieved_results"], 1):
    print(f"\n--- Result {i} ---")
    print("Customer:", item["customer_query"])
    print("Support:", item["agent_response"])

print("\n" + "=" * 60)