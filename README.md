# End-to-End Automated QA Generation Agent

## Abstract

- **System Overview**: Designed and implemented an end-to-end automated QA generation system for training and deploying a chatbot for students, faculty, and staff across campus, significantly enhancing information accessibility throughout the university ecosystem.
- **Modular Architecture**: Devised a modular architecture featuring domain-optimized web crawling with bot detection avoidance (user-agent rotation, request delays, persistent checkpointing), semantic chunking with Named Entity Recognition (spaCy/NLTK) and key phrase extraction (TF-IDF/noun phrase detection), and Retrieval-Augmented Generation featuring cross-encoder reranking to power Large Language Models for precise, context-aware question–answer generation.
- **Mixture of Experts Architecture**: Engineered a Mixture of Experts architecture integrating DeepSeek R1, Llama, Flan-T5 models (XL/XXL/Large/Base), cross-encoders (ms-marco-MiniLM-L-12-v2/L-6-v2), sentence transformers (all-mpnet-base-v2, all-MiniLM-L12-v2), and fact-checking models (vectara/hallucination_evaluation) with adaptive cascading to optimize resource utilization while boosting contextual relevance and semantic coherence in question–answer generation.
- **Efficiency**: Achieved remarkable efficiency, generating 300 high-quality QA pairs by crawling 30 webpages in under ten minutes using an A100 GPU, demonstrating the system's production readiness and real-time capability for continuous content updates.
- **Quality Control**: Developed sophisticated quality control mechanisms with multi-dimensional scoring (relevance, factuality, completeness, formatting) using hybrid metrics—combining n-gram overlap, semantic similarity, and specialized models for hallucination detection, toxicity classification, and Personally Identifiable Information detection—to ensure high-quality, trustworthy responses that uphold standards of fairness, privacy protection, and content inclusivity, with minimal fabrication.
- **Chain-of-Thought Prompting**: Devised chain-of-thought prompting to produce reasoning-oriented questions requiring step-by-step analysis, thereby enhancing the chatbot's multi-hop reasoning capabilities for university-specific inquiries (e.g., policies, resources, services).

## Features

- **Intelligent Web Crawler**: Domain-optimized with bot detection avoidance, request delays, and checkpointing
- **Advanced Document Processing**: Semantic chunking with NER and phrase extraction
- **Knowledge Base**: Efficient storage and retrieval of processed content
- **RAG-Powered Generation**: Using state-of-the-art language models with cross-encoder reranking
- **Adaptive Model Selection**: Cascading fallbacks for optimal resource utilization
- **Quality Control**: Multi-dimensional scoring for high-quality QA pairs
- **Chain-of-Thought Prompting**: For complex reasoning questions
- **Checkpointing**: Resume operations from previous runs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qa-generation-system.git
cd qa-generation-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK and spaCy resources
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"
python -m spacy download en_core_web_sm
```

## Usage

```bash
python qa_generator.py --url "https://www.example.edu" --output "qa_output" --max-pages 30 --max-pairs 10
```


## Output Structure

The system produces output files in the specified output directory (--output), including:

- qa_pairs_final.json: JSON file containing all QA pairs with metadata
- qa_pairs_final.csv: CSV file with columns:
-- Question
-- Answer
-- Source URL
-- Topic
-- Topic Type
-- Overall Score
- qa_pairs_final.md: Markdown file grouping QA pairs by source URL with a hierarchical structure
- 
Checkpoint files are stored in the checkpoint directory (--checkpoint):

- progress_{hash}.json: Tracks overall progress
- content_{hash}.json: Cached web content
- visited_{hash}.json: List of visited URLs
- failed_{hash}.json: List of failed URLs
- scores_{hash}.json: Quality scores for pages
- kb_{hash}.json: Processed knowledge base
- qa_pairs_{hash}.json: Intermediate QA pairs

## Example Execution
Here are several example commands demonstrating different use cases:
```bash
# Basic execution with default settings
python qa_generator.py --url "https://www.studentservices.university.edu" --output "university_qa"

# High-volume crawling with increased QA pairs
python qa_generator.py --url "https://www.registrar.university.edu" --output "registrar_qa" --max-pages 50 --max-pairs 15

# Running without GPU support
python qa_generator.py --url "https://www.financialaid.university.edu" --output "financial_aid_qa" --no-gpu

# Forcing regeneration despite existing checkpoints
python qa_generator.py --url "https://www.housing.university.edu" --output "housing_qa" --force
```
