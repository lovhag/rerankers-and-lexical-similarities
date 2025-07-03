import os
import pandas as pd
from tqdm import tqdm
from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter(language='en')

def merge_erroneous_newlines(paragraphs):
    PUNCTUATIONS = ['!', '"', '#', '$', '%', "'", ')', '.', '/', ':', ';', 
                '<', '>', '?', '\\', ']', '^', '`', '|', '}', '~']
    def paragraph_is_interrupted(paragraph, next_paragraph):
        next_paragraph = next_paragraph.strip()
        if len(next_paragraph) > 0:
            return (paragraph[-1] not in PUNCTUATIONS) and (not next_paragraph[0].isupper())
        else:
            return True
    
    if len(paragraphs) < 2:
        return paragraphs

    curr_paragraph = paragraphs[0]
    new_paragraphs = []
    for paragraph in paragraphs[1:]:
        if paragraph_is_interrupted(curr_paragraph, paragraph):
            curr_paragraph = (" ").join([curr_paragraph.strip(), paragraph])
        else:
            new_paragraphs.append(curr_paragraph)
            curr_paragraph = paragraph
    new_paragraphs.append(curr_paragraph.strip())
    return new_paragraphs

def preprocess_content(content):
    # reformat quotes
    content = content.replace('“', '"').replace('”', '"')
    return content

def keep_paragraph(s, min_word_limit):
    if len(s.split()) < min_word_limit:
        return False
    return True

def split_paragraph(s, max_word_limit):
    parts = splitter.split(text=s)
    words_per_part = [len(p.split()) for p in parts]
    
    cum_words = 0
    for i, l in enumerate(words_per_part):
        cum_words += l
        if cum_words > max_word_limit:
            split_ix = i
            break
    if split_ix == 0:
        # if there is no good split candidate, just cut the first part
        first_parts = (" ").join(parts[0].split()[:max_word_limit])
        last_parts = (" ").join([(" ").join(parts[0].split()[max_word_limit:])]+parts[1:])
    else: 
        first_parts = (" ").join(parts[:split_ix])
        last_parts = (" ").join(parts[split_ix:])
    return first_parts, last_parts

def split_paragraphs(paragraphs, max_word_limit):
    new_paragraphs = []
    for s in paragraphs:
        s_parts = [s]
        while len(s_parts[-1].split()) > max_word_limit:
            first_parts, last_parts = split_paragraph(s_parts[-1], max_word_limit)
            # stop if the content is simply too long
            if len(last_parts) > 100000:
                return None
            s_parts[-1] = first_parts
            s_parts.append(last_parts)
        new_paragraphs.extend(s_parts)
    return new_paragraphs

def get_paragraphs_from_crawl(crawl_results_path, save_folder, min_word_limit, max_word_limit):
    crawl_results = pd.read_json(crawl_results_path, lines=True)
    print(f"Read {len(crawl_results)} search result samples from '{crawl_results_path}'.")
    print()
        
    # crawl pages
    print("Extracting paragraphs from the crawled content:")
    content_paragraphs = []
    # TODO: process only content per unique link? (now there may be duplicates)
    for _, row in tqdm(crawl_results.iterrows(), total=len(crawl_results)):
        content = preprocess_content(row.content)
        paragraphs = content.splitlines()
        paragraphs = merge_erroneous_newlines(paragraphs)
            
        # remove too short paragraphs
        paragraphs = [s for s in paragraphs if keep_paragraph(s, min_word_limit)]        
        # split too long paragraphs
        paragraphs = split_paragraphs(paragraphs, max_word_limit)     
        # remove sentence duplicates
        if paragraphs is not None and len(paragraphs) > 0:
            paragraphs = pd.Series(paragraphs).drop_duplicates().to_list()   
            content_paragraphs.append(paragraphs)
        else: 
            content_paragraphs.append(None)
    
            
    crawl_results["paragraphs"] = content_paragraphs
    print("Paragraph splitting done!")
    print()

    is_empty_mask = crawl_results.paragraphs.isna()
    print(f"Found {sum(is_empty_mask)} entries with empty or too short content.")
    failed_filepath = os.path.join(save_folder, "paragraph_problem.csv")
    # Store the samples to allow for further analysis
    print(f"Storing failed examples to '{failed_filepath}'.")
    crawl_results[is_empty_mask].to_csv(failed_filepath)
    
    print("Dropping the erronous samples")
    crawl_results = crawl_results[~(is_empty_mask)]
    print(f"{len(crawl_results)} samples remain.")
    print()
    
    save_file = os.path.join(save_folder, "paragraph_results.jsonl")
    crawl_results.to_json(save_file, orient='records', lines=True)
    print(f"Done! A total of {len(crawl_results)} paragraphed content results have been saved to '{save_file}'.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Structure crawled results into paragraphs script")
    parser.add_argument("--crawl_results_path", type=str, required=True, help="Path to the crawl results CSV file")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save the data to")
    parser.add_argument("--min_word_limit", type=int, default=4, help="Min word limit for a paragraph to be kept")
    parser.add_argument("--max_word_limit", type=int, default=200, help="Max word limit for a paragraph not to be splitted")
    
    args = parser.parse_args()
    
    results_file = os.path.join(args.save_folder, "paragraph_results.csv")
    if os.path.isfile(results_file):
        print(f"A results file already exists for this script. ('{results_file}')")
        print("Remove it if you want to rerun the script.")
    else:
        os.makedirs(args.save_folder, exist_ok=True)
        get_paragraphs_from_crawl(args.crawl_results_path, args.save_folder, args.min_word_limit, args.max_word_limit)