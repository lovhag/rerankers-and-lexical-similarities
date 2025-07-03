import argparse
import pandas as pd
import bert_score
from tqdm import tqdm

tqdm.pandas()

def get_cands_and_refs(row):
    # if NQ data with chunks formatted as lists of dicts
    if isinstance(row.chunks[0], dict):
        refs = [val["chunk"] for val in row.chunks]
    # if LitQA2 with chunks simply given by a list of strings
    else:
        refs = row.chunks
    return pd.Series({'cands': [row.question]*len(row.chunks),
            'refs': refs,
            'id': row.id})

def main():
    parser = argparse.ArgumentParser(description="Calculate BERT scores for samples from a given dataset.")
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
    
    # Prepare the dataset for more efficient batching wrt BERT scores
    eval_data = pd.DataFrame(data.apply(get_cands_and_refs, axis=1)).explode(['cands', 'refs'])
    
    # Calculate BERT scores
    def get_score(cands, refs):
        _, _, f1 = bert_score.score(cands, refs, lang='en', batch_size=256, verbose=True)
    
        return [val.item() for val in f1]
    eval_data['bert_score'] = get_score(eval_data.cands.tolist(), eval_data.refs.tolist())
    
    # implode dataset to old format
    eval_data = (eval_data.groupby('id')
      .agg({'bert_score': lambda x: x.tolist()})
      .reset_index())
    data = data.set_index('id')
    eval_data = eval_data.set_index('id')
    data = data.merge(eval_data.bert_score, on='id').reset_index().drop(columns='id')
    
    # Store the results
    save_file = args.data_file.replace(".jsonl", "_bert_scores.jsonl")
    data.to_json(save_file, orient='records', lines=True)
    print(f"BERT scores processed and saved to '{save_file}'.")

if __name__ == "__main__":
    main()