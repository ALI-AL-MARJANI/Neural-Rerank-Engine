# Neural Rerank Engine

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)


---

## Overview

Standard Vector Search (RAG) often suffers from a "precision" problem: it retrieves documents that are *semantically* similar but factually irrelevant. Keyword search (BM25) is precise but misses context.

**Neural-Rerank-Engine** solves this by implementing an industry-standard **Retrieve & Rerank** architecture:

1.  **Stage 1 (Retrieval):** Fast retrieval of the top 50 candidates using a **Hybrid** approach (Sparse BM25 + Dense Bi-Encoder Embeddings).
2.  **Stage 2 (Reranking):** A heavy **Cross-Encoder (BERT)** inspects the query-document pairs to re-order results with high semantic understanding.

### Key Objectives
* **Maximize Recall:** Use Hybrid Search to ensure the correct answer is *somewhere* in the top candidates.
* **Maximize Precision:** Use Reranking to push the best answer to the top 1.
* **Latency Optimization:** Balance the speed of FAISS with the accuracy of BERT.

---

## Architecture


<div align="center">
  <img src="assets/archi.png" alt="Architecture" width="100%" style="border-radius: 10px; border: 1px solid #e1e4e8;">
</div>
<br>



