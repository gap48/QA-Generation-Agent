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
