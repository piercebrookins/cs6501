from __future__ import annotations

# Exercise 1 queries
MODEL_T_QUERIES: list[str] = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]

CONGRESSIONAL_RECORD_QUERIES: list[str] = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

EX1_ALL_QUERIES: list[str] = [*MODEL_T_QUERIES, *CONGRESSIONAL_RECORD_QUERIES]

# Exercise 5
UNANSWERABLE_OFF_TOPIC: list[str] = [
    "What is the capital of France?",
    "Who won the 2024 Super Bowl?",
]

UNANSWERABLE_RELATED_BUT_NOT_IN_CORPUS: list[str] = [
    "What's the horsepower of a 1925 Model T?",
    "What is the fuel injection system spec?",
]

UNANSWERABLE_FALSE_PREMISE: list[str] = [
    "Why does the manual recommend synthetic oil?",
    "What does the manual say about electronic ignition?",
]

# Exercise 6: 5+ phrasings of one question
PHRASING_SENSITIVITY_QUESTION_VARIANTS: list[str] = [
    "What is the recommended maintenance schedule for the engine?",
    "How often should I service the engine?",
    "engine maintenance intervals",
    "When do I need to check the engine?",
    "Preventive maintenance requirements",
    "How frequently should engine maintenance be performed?",
]

# Exercise 9 (sample 10 diverse queries for Model T-like manuals)
SCORE_ANALYSIS_QUERIES: list[str] = [
    "How do I adjust the carburetor?",
    "What oil should I use?",
    "How to fix the transmission?",
    "What is the spark plug gap?",
    "Engine maintenance schedule",
    "Tire pressure recommendations",
    "How to start the engine in cold weather?",
    "Brake adjustment procedure",
    "Cooling system maintenance",
    "Electrical system troubleshooting",
]

# Exercise 10 prompt templates
TEMPLATE_MINIMAL = """Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"""

TEMPLATE_STRICT = """You are a helpful assistant. Answer ONLY based on the
provided context. If the answer is not in the context, say \"I cannot answer
this from the available documents.\"

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

TEMPLATE_CITATION = """You are a research assistant. Answer the question using
ONLY the provided context. Quote the exact passages that support your answer.
Cite the source file for each quote.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""

TEMPLATE_PERMISSIVE = """You are a knowledgeable assistant. Use the provided
context to help answer the question. You may also use your general knowledge
to supplement the answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

TEMPLATE_STRUCTURED = """You are an analytical assistant.

CONTEXT:
{context}

QUESTION: {question}

First, list the relevant facts from the context.
Then, synthesize your answer.

RELEVANT FACTS:
SYNTHESIS:"""

PROMPT_TEMPLATES: dict[str, str] = {
    "minimal": TEMPLATE_MINIMAL,
    "strict": TEMPLATE_STRICT,
    "citation": TEMPLATE_CITATION,
    "permissive": TEMPLATE_PERMISSIVE,
    "structured": TEMPLATE_STRUCTURED,
}

# Exercise 11: failure modes
FAILURE_MODE_QUERIES: list[tuple[str, str]] = [
    ("Computation", "What is the total weight of all parts listed?"),
    ("Temporal reasoning", "Which maintenance task should be done first?"),
    ("Comparison", "Is procedure A easier than procedure B?"),
    ("Ambiguous", "How do you fix it?"),
    ("Multi-hop", "What tool is needed for the part that connects to the carburetor?"),
    ("Negation", "What should you NOT do when adjusting the timing?"),
    ("Hypothetical", "What would happen if I used the wrong oil?"),
]

# Exercise 12: cross-document synthesis
CROSS_DOC_QUERIES: list[str] = [
    "What are ALL the maintenance tasks I need to do monthly?",
    "Compare the procedures for adjusting X vs. adjusting Y",
    "What tools do I need for a complete tune-up?",
    "Summarize all safety warnings in the manual",
]
