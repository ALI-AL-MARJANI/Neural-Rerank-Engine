import json
from datasets import load_dataset

def generate_test_set(output_path="data/test_queries.json", num_queries=50):
    """
    Extracts queries and their corresponding document IDs to create a 'Gold Standard' test set
    """
    
    
   
    dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    
    test_set = []
    
    for i, row in enumerate(dataset):
        if i >= num_queries:
            break
            
        # In our dataset, the query_id corresponds to the doc_id of the relevant document
        # So the 'correct answer' for this query is this doc_id.
        query_id = str(row['query_id'])
        query_text = row['query']
        
        test_set.append({
            "query_id": query_id,
            "query": query_text,
            "expected_doc_id": query_id  # This is the target we want to retrieve
        })

    # Save the test set to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=4)
        
    print(f" Test set generated with {len(test_set)} queries in {output_path}")

if __name__ == "__main__":
    generate_test_set()