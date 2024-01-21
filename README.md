
# Implementing a RAG Pipeline on the Wikipedia Dataset

## Overview
This Jupyter notebook demonstrates the implementation of a Retrieval-Augmented Generation (RAG) pipeline using the Wikipedia dataset. The primary goal is to showcase how large datasets can be efficiently processed and utilized in NLP tasks.

### Key Features
- **Efficient Data Handling**: Utilizes memory mapping between RAM and filesystem storage via the Hugging Face Datasets library, leveraging the Apache Arrow format and pyarrow library.
- **Embedding with Sentence Transformers**: Employs the `all-MiniLM-L6-v2` model from the Sentence-Transformers library for embedding Wikipedia articles into a 384-dimensional vector space.
- **Similarity Search with Faiss**: Implements similarity searches using the `faiss.IndexFlatL2` index based on Euclidean (L2) distance.
- **Multi-GPU Processing**: Optimized to run on multiple GPUs, specifically 2xT4 GPUs provided by Kaggle.
- **Question Answering Pipeline**: Uses the `distilbert-base-cased-distilled-squad` Q&A pipeline for answering questions based on the embedded Wikipedia dataset.

### Technologies Used
- Hugging Face Datasets
- BlingFire
- Sentence-Transformers
- Faiss (Facebook AI Similarity Search)
- DistilBERT
- PyTorch
- Kaggle GPUs

## Getting Started

### Prerequisites
- A machine with at least 2xT4 GPUs (for optimal performance).
- Python 3.x with the following libraries installed:
  - Hugging Face Datasets
  - Sentence-Transformers
  - Faiss
  - PyTorch

### Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/emmermarcell/wikipedia-rag-pipeline
cd wikipedia-rag-pipeline
pip install -r requirements.txt
```

### Running the Notebook
- Open the notebook in a Jupyter environment.
- Ensure that your machine is configured to use the GPUs.
- Run the cells in order to process the Wikipedia dataset and perform the RAG pipeline tasks.

## Usage
The notebook is divided into several sections, each handling different aspects of the RAG pipeline:
1. **Data Loading and Preprocessing**: How to load and preprocess the Wikipedia dataset.
2. **Embedding Articles**: Instructions on embedding article text using Sentence Transformers.
3. **Similarity Search**: Steps to perform similarity searches with Faiss.
4. **Question Answering**: Utilizing the DistilBERT Q&A pipeline to answer questions based on the processed data.

### Example Queries
You can test the system with business-related questions such as:
- "What services does KPMG offer to its clients?"
- "How do you stay updated on changes in tax laws?"

## Acknowledgements
Special thanks to the authors of the following resources for their insights and methodologies which greatly influenced this implementation:
- Implementing RAG with Langchain and Hugging Face by Akriti Upadhyay
- Ask Wikipedia ELI5-like Questions Using Long-Form Question Answering on Haystack by Vladimir Blagojevic
- Pre-processing a Wikipedia dump for NLP model training by Steven van de Graaf

---

