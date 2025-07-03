import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import torch

from src.collect_metrics.get_bert_scores import get_cands_and_refs

tqdm.pandas()

def main():
    parser = argparse.ArgumentParser(description="Calculate reranker scores for samples from a given dataset.")
    parser.add_argument("--data_file", type=str, help="Path to the input file containing samples to process (should have the columns 'question' and 'chunks').")
    
    args = parser.parse_args()
    
    # Read the dataset from the input file
    data = pd.read_json(args.data_file, lines=True)
    # Get a temporary id field for later use when unwrapping the dataset
    if 'NQ' in args.data_file:
        id_field = 'nq_example_id'
    elif 'DRUID' in args.data_file:
        id_field = 'claim_id'
    elif 'LitQA2' in args.data_file:
        id_field = 'id'
    data['id'] = data[id_field].copy()
    
    # Prepare the dataset for more efficient batching
    eval_data = pd.DataFrame(data.apply(get_cands_and_refs, axis=1)).explode(['cands', 'refs'])
    
    # initialise the reranker
    model = AutoModelForSequenceClassification.from_pretrained('jinaai/jina-reranker-v1-turbo-en', num_labels=1, trust_remote_code=True, cache_dir="data/cache")
    
    # Calculate ranks
    eval_data['jina_reranker_v1_turbo_en_rank'] = model.compute_score([[eval_data.cands.iloc[ix], eval_data.refs.iloc[ix]] for ix in range(len(eval_data.cands))], batch_size=128, device=torch.device('cuda'))
    
    # implode dataset to old format
    eval_data = (eval_data.groupby('id')
      .agg({'jina_reranker_v1_turbo_en_rank': lambda x: x.tolist()})
      .reset_index())
    data = data.set_index('id')
    eval_data = eval_data.set_index('id')
    data = data.merge(eval_data.jina_reranker_v1_turbo_en_rank, on='id').reset_index().drop(columns='id')
    
    # Store the results
    save_file = args.data_file.replace(".jsonl", "_jina_reranker_v1_turbo_en_rank.jsonl")
    data.to_json(save_file, orient='records', lines=True)
    print(f"Ranks processed and saved to '{save_file}'.")

if __name__ == "__main__":
    main()