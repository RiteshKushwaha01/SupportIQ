# Hiver Support Agent

An AI-powered support agent that understands customer support queries, retrieves relevant information from a knowledge base, and generates helpful responses.

## Overview

The Hiver Support Agent is designed to automate and assist with customer support workflows.

The system follows a retrieval-augmented generation (RAG) approach:

Customer Query
        ↓
Query Classification
        ↓
Relevant Information Retrieval
        ↓
Context + Query
        ↓
LLM Response Generation
        ↓
Support Response

## Goals

- Understand customer support queries
- Retrieve relevant information from historical support cases and knowledge sources
- Generate accurate and contextual responses
- Reduce repetitive manual support work
- Provide a foundation for automated ticket handling

## Project Structure

```text
hiver-support-agent/
│
├── .venv/                         # Python virtual environment
├── .env                           # Environment variables
├── .gitignore
├── README.md
├── requirements.txt
├── run.py                         # Application entry point
│
├── app/
│   ├── __init__.py
│   ├── config.py                  # Application configuration
│   ├── models.py                  # Data models / schemas
│   ├── prompts.py                 # LLM prompts
│   │
│   └── services/
│       ├── __init__.py
│       ├── classifier.py           # Support query classification
│       ├── retriever.py            # Relevant information retrieval
│       ├── response_generator.py   # Generate support responses
│       └── ticket_service.py       # Ticket creation/update logic
│
├── data/
│   ├── raw/                       # Raw support data
│   ├── processed/                 # Cleaned/processed data
│   └── corpus/                    # Final retrieval corpus
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_build_retrieval_corpus.ipynb