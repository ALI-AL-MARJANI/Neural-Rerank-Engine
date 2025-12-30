# Neural Rerank Engine

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green?style=for-the-badge)


---

## Overview

Standard Vector Search (RAG) often suffers from a "precision" problem: it retrieves documents that are *semantically* similar but factually irrelevant. Keyword search (BM25) is precise but misses context.

**Neural-Rerank-Engine** solves this by implementing a standard **Retrieve & Rerank** architecture:

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

## Performance Benchmarks
We evaluated the system on a subset of the MS MARCO dataset. The results demonstrate a significant precision boost using the hybrid approach:

| Method | MRR@10 (Accuracy) | Avg Latency |
| :--- | :--- | :--- |
| **BM25 (Sparse)** | 0.5594 | ~7 ms |
| **FAISS (Dense)** | 0.7977 | ~64 ms |
| **Hybrid + Rerank** | **0.8282** | ~165 ms |

---

## Future Work & State of the Art

While this project implements the current industry standard (Bi-Encoder + Cross-Encoder), research in Neural Information Retrieval is moving fast.

**Research Insight (2025): Listwise Reranking with LLMs** :

A promising direction to reduce the computational cost of Cross-Encoders is explored in the paper "LLMs can reason over BM25 scores to Improve Listwise Reranking" (https://arxiv.org/pdf/2506.14086).

The paper suggests that instead of feeding full document text to a heavy model, we can feed retrieval scores and metadata to a lightweight LLM. The LLM can "reason" about the distribution of scores to re-rank documents efficiently.









