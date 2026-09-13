from src.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()

prompt, results = pipeline.build_context(
    "My Fire TV Stick is not working",
    3
)

print("\nRetrieved:", len(results))

print("\nTop result:")
print(results[0])

print("\nPrompt created successfully!")