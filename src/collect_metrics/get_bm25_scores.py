import argparse
import pandas as pd
from tqdm import tqdm

import bm25s
import Stemmer

tqdm.pandas()

def main():
    parser = argparse.ArgumentParser(description="Calculate BM25 similarity scores for samples from a given dataset.")
    parser.add_argument("--data_file", type=str, help="Path to the input file containing samples to process (should have the columns 'question' and 'chunks').")
    
    args = parser.parse_args()
    
    # Read the dataset from the input file
    data = pd.read_json(args.data_file, lines=True)
    
    # Calculate BM25 scores
    stemmer = Stemmer.Stemmer("english")
    def get_bm25_score(row):
        if isinstance(row.chunks[0], dict):
            corpus = [val["chunk"] for val in row.chunks]
        else:
            corpus = row.chunks
        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
        
        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # Query the corpus
        query_tokens = bm25s.tokenize(row.question, stemmer=stemmer)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
        results, scores = retriever.retrieve(query_tokens, k=len(corpus))
        sim_scores = [None]*len(corpus)
        for i in range(results.shape[1]):
            sim_scores[results[0, i]] = scores[0, i]
        
        return sim_scores
        
    data["bm25_score"] = data.progress_apply(get_bm25_score, axis=1)
    
    # Store the results
    save_file = args.data_file.replace(".jsonl", "_bm25_scores.jsonl")
    data.to_json(save_file, orient='records', lines=True)
    print(f"BM25 scores processed and saved to '{save_file}'.")

if __name__ == "__main__":
    main()