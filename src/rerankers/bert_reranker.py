from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class BertReranker:
    """
    Re-ranks a list of candidate documents using a Cross-Encoder model.
    This is slower but much more accurate than vector search.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Scores and re-sorts the documents.
    
        """
        if not docs:
            return []

        # Prepare pairs for the model: 
        pairs = [[query, doc['text']] for doc in docs]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Attach scores to documents
        for doc, score in zip(docs, scores):
            doc['rerank_score'] = float(score)

        # Sort by new score (Descending)
        sorted_docs = sorted(docs, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_docs[:top_n]