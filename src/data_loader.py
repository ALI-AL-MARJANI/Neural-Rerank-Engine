import os
import json
from datasets import load_dataset

def load_msmarco_subset(output_path="data/raw/corpus.json", max_samples=10000):
    """
    Downloads a subset of the MS MARCO dataset (Passage Ranking).
    Uses 'streaming' mode to avoid downloading the full 50GB dataset.
    
    Args:
        output_path (str): Path to save the JSON file.
        max_samples (int): Number of passages to retrieve.
    """
    
    
    # Load dataset in streaming mode 
    try:
        dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset via Hugging Face: {e}")
        return []
    
    data = []
    
    # Retrieve only the first N examples
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
            
        # MS MARCO structure: 
        # row['passages'] is a dictionary with 'passage_text' (list)
        # We take the first available passage for simplicity
        try:
            text = row['passages']['passage_text'][0]
            doc_id = row['query_id'] # Use query_id as a temporary doc_id
            
            data.append({
                "id": str(doc_id),
                "text": text,
                "metadata": {"source": "ms_marco_v1.1"}
            })
        except (KeyError, IndexError):
            continue
        
        if i > 0 and i % 1000 == 0:
            print(f"   ... {i} passages processed")

    # Save locally
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f" Done! {len(data)} documents saved to {output_path}")
    return data

if __name__ == "__main__":
    load_msmarco_subset()