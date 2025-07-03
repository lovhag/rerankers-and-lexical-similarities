# rerankers-and-lexical-similarities

This is the repo for the paper ["Language Model Re-rankers are Fooled by Lexical Similarities"](https://arxiv.org/abs/2502.17036) accepted to FEVER 2025.

<p align="center">
  <img src=https://github.com/user-attachments/assets/6add9a52-6426-46d8-88ec-159e5b5033a2 alt="An overview of a RAG pipeline." style="width:30%; height:auto;">
</p>

The paper is based on two main contributions:
1. the collection of samples from three diverse datasets that test the performance of re-rankers, and 
2. careful evaluation of re-ranker performance based on novel metrics that compare re-ranker performance to a BM25 baseline. 

The code underlying these contributions is described below.

## Datasets for re-ranker evaluation

We collect samples from [NQ](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question), LitQA2 (from [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench)) and [DRUID](https://huggingface.co/datasets/copenlu/druid) for the evaluation of re-rankers. The samples are collected using the notebook [src/gather_data.ipynb](src/gather_data.ipynb).

We have also uploaded the processed evaluation datasets to [Hugging Face datasets](https://huggingface.co/datasets/Lo/rerankers-and-lexical-similarities), so one doesn't have to run the notebook mentioned above to get the evaluation datasets.

## Evaluation of re-rankers

We first collect our metric of interest for the dataset samples using [src/collect_metrics.ipynb](src/collect_metrics.ipynb). Our metrics of interest are:
- Various similarity scores (BERT scores, Jaccard similarities, BM25 scores)
- Various re-ranker scores (Cohere re-ranker, BGE re-ranker, and Jina re-rankers) 
 
We also collect re-ranker scores from GPT-4o m and GPT-4o. This approach is described in Appendix E in our paper.

Finally, the results are plotted in [src/evaluate.ipynb](src/evaluate.ipynb). For the analysis in the notebook, we assume that the data files with all relevant metrics have been prepared as described above.

## Citation

```
@misc{languagemodelrerankersfooled,
      title={Language Model Re-rankers are Fooled by Lexical Similarities}, 
      author={Lovisa Hagström and Ercong Nie and Ruben Halifa and Helmut Schmid and Richard Johansson and Alexander Junge},
      year={2025},
      eprint={2502.17036},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.17036}, 
}
```