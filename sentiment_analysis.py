import json
import tqdm
import logging

import numpy as np

from collections import defaultdict
from allennlp.predictors.predictor import Predictor

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)

SENT_MODEL_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/sst-2-basic-classifier-glove-2019.06.27.tar.gz"


def main():
    sentiment_analyzer = Predictor.from_path(SENT_MODEL_PATH)

    model_names = {'gpt2': 'GPT2-small', 'gpt2-large': 'GPT2-large', 'gpt2-medium': 'GPT2-medium',
                   'gpt2-xl': 'GPT2-XL', 'openai-gpt': 'GPT', 'transfo-xl-wt103': 'TransformerXL',
                   'xlnet-base-cased': 'XLNet-base', 'xlnet-large-cased': 'XLNet-large'}

    most_negative = {}
    sentiment_by_name = {}

    for model_name, display_name in model_names.items():
        sentiment_by_name[display_name] = get_sentiment(sentiment_analyzer, model_name)
        most_negative[display_name] = most_negative_people(sentiment_by_name[display_name])

    # Print the results
    names = [json.loads(line.strip()) for line in open("../names.jsonl")]
    news_names = set([e["name"] for e in names if "news" in e["attributes"]])
    to_latex_table(most_negative, news_names)


def get_sentiment(sentiment_analyzer, model_name):
    """
    Get the sentiment (positive score, negative score) for every names for a given LM
    """
    endings = [json.loads(line.strip()) for line in open(f'../is_a_endings/{model_name}.jsonl')]
    sentiment_by_name = defaultdict(list)
    for example in tqdm.tqdm(endings):
        sentiment = sentiment_analyzer.predict(sentence=example['text'])
        sentiment_by_name[example['name']].append(sentiment['probs'])

    return sentiment_by_name


def most_negative_people(model_sentiment_by_name):
    """
    Rank the people by their average negative score for a given LM
    """
    negative_sentiments = {name: (np.mean([s[1] for s in predictions]), np.std([s[1] for s in predictions]))
                           for name, predictions in model_sentiment_by_name.items()}

    most_negative = sorted(negative_sentiments.items(), key=lambda x: x[1][0], reverse=True)
    return most_negative


def to_latex_table(results, news_names, k=10):
    """
    Prints a latex table with the most negative sentiment results.
    Don't forget: usepackage{booktabs}, usepackage{multirow}
    """
    print("""\\begin{tabular}{l l l l l l l l l l l l l l l l}""")
    print("""\\toprule""")

    order = ['GPT', 'GPT2-small', 'GPT2-medium', 'GPT2-large', 'GPT2-XL',
             'TransformerXL', 'XLNet-base', 'XLNet-large']
    print(" & ".join(["\multicolumn{2}{c}{\\textbf{" + model_name + "}}"
                      for model_name in order]), end=" \\\\ \n")

    for i in range(1, 16, 2):
        print("\cmidrule(lr){" + str(i) + "-" + str(i + 1) + "}")

    s = """\\textbf{Name} & $\mathbf{F_1}$"""
    print(" & ".join([s] * len(order)), end=" \\\\ \n")
    print('\midrule')

    top_k = {model_name: sorted(curr_results, key=lambda x: x[1], reverse=True)[:k]
             for model_name, curr_results in results.items()}

    for i in range(k):
        for idx, model_name in enumerate(order):
            name, score = top_k[model_name][i]
            display_name = "\\textbf{" + name + "}" if name in news_names else name
            print(display_name + f" & {score[0]:.3f}", end=' & ' if idx < len(model_names) - 1 else " \\\\ ")
        print('')

    print("""\\bottomrule""")
    print("""\end{tabular}""")


if __name__ == "__main__":
    main()
