# Named Entity Biases in Pre-trained Language Models

This repository contains the experiments used in the paper:

**"You are grounded!": Latent Name Artifacts in Pre-trained Language Models.** *Vered Shwartz, Rachel Rudinger, and Oyvind Tafjord*. arXiv 2020. 

### 1. Last Name Prediction:

Run the script `predict_last_name.sh`. It will produce the file `data/all_names_results.tsv`. 

### 2. Given Name Recovery:

Run the script `predict_given_name.sh [device]` with a GPU number of "cpu". It will save the results in a json file for each language model under `results`.

### 3. Sentiment Analysis:

Run the script `sentiment_analysis.py`. It will produce the LaTex table with the results. 

### 4. Effect on Downstream Tasks:

The `downstream` directory contains the templates, sampled names to assign to the templates, and a notebook to run the experimets for `Winogrande` and `SQuAD`. 


### References 

Please cite this repository using the following reference:

```
@inproceedings{Bosselut2019COMETCT,
  title={``You are grounded!'': Latent Name Artifacts in Pre-trained Language Models},
  author={Vered Shwartz and Rachel Rudinger and Oyvind Tafjord},
  booktitle={arXiv},
  year={2020}
}
```
