import math
import json
import tqdm
import torch
import logging
import argparse
import itertools
import numpy as np

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from collections import defaultdict, Counter
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="LM name")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help="CPU/GPU device")
    parser.add_argument("--out_dir", default='results', type=str,
                        required=False, help="The results directory. Saves one file per model.")
    parser.add_argument("--include_bert", action="store_true", help="include a BERT-based classifier.")
    args = parser.parse_args()
    logger.debug(args)

    with open(f'{args.out_dir}/{args.model_name}.jsonl', 'w') as f_out:
        for gender, name_repl in [('male', 'David'), ('female', 'Mary')]:
            for curr_names, train, val, test in split_to_train_val_test(args.model_name, [gender], name_repl=name_repl):
                (train_X, train_y), (val_X, val_y), (test_X, test_y) = zip(*train), zip(*val), zip(*test)

                majority_test_predictions, p, r, f1 = majority_baseline(train_X, train_y, val_X, val_y, test_X, test_y)
                out = json.dumps({"gender": gender, "method": 'Majority', "name1": curr_names[0],
                                  "name2": curr_names[1], "precision": p, "recall": r, "f1": f1})
                logger.info(out)
                f_out.write(out + '\n')
                f_out.flush()

                svm_test_predictions, p, r, f1, setup = svm_with_tfidf(train_X, train_y, val_X, val_y, test_X, test_y)
                out = {"gender": gender, "method": 'SVM + tf-idf', "name1": curr_names[0],
                       "name2": curr_names[1], "precision": p, "recall": r, "f1": f1}
                out.update(setup)
                out = json.dumps(out)
                logger.info(out)
                f_out.write(out + '\n')
                f_out.flush()

                if args.include_bert:
                    bert_model = FineTuneBert(
                        'bert-large-uncased', args.device, train_X, train_y, val_X, val_y, test_X, test_y)
                    bert_test_predictions, p, r, f1, setup = bert_model.fine_tune()
                    out = {"gender": gender, "method": 'BERT', "name1": curr_names[0],
                           "name2": curr_names[1], "precision": p, "recall": r, "f1": f1}
                    out.update(setup)
                    out = json.dumps(out)
                    logger.info(out)
                    f_out.write(out + '\n')
                    f_out.flush()


def split_to_train_val_test(model_name, attributes, name_repl=None):
    """
    For a given model name (e.g. "gpt2-xl") and attributes (e.g. ["male"])
    Get all the endings of items with these attributes labeled with their names.
    Split to train (80%), test (10%), and validation(10%).
    If name_repl is not None, [NAME] will be replaced by this string.
    """
    all_endings = [json.loads(line.strip()) for line in open(f'is_a_endings/{model_name}.jsonl')]
    all_endings = [e for e in all_endings if set(attributes).issubset(set(e["attributes"]))]

    by_name = defaultdict(list)
    [by_name[e["name"]].append(e["text"]) for e in all_endings]

    for name1, name2 in itertools.combinations(by_name.keys(), 2):
        train, val, test = [], [], []
        curr_data = {name: by_name[name] for name in [name1, name2]}

        for name, texts in curr_data.items():
            if name_repl is not None:
                texts = [text.replace("[NAME]", name_repl) for text in texts]

            train_size = int(math.ceil(len(texts) * 0.8))
            val_size = int(math.ceil(len(texts) * 0.1))
            train.extend([(text, name) for text in texts[:train_size]])
            val.extend([(text, name) for text in texts[train_size:train_size + val_size]])
            test.extend([(text, name) for text in texts[train_size + val_size:]])

        logging.info(f'Train size: {len(train)}, val size: {len(val)}, test size: {len(test)}')

        assert (len(set(train).intersection(set(val))) == 0)
        assert (len(set(train).intersection(set(test))) == 0)
        assert (len(set(test).intersection(set(val))) == 0)

        yield (name1, name2), train, val, test


def majority_baseline(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    Test predictions, precision, recall and F1 scores for the majority baseline.
    """
    majority_label = Counter(train_y).most_common(1)[0][0]
    test_predictions = [majority_label for _ in test_y]
    p, r, f, _ = precision_recall_fscore_support(test_y, test_predictions, average='weighted')
    return test_predictions, p, r, f


def svm_with_tfidf(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    Represent the texts with tf-idf and train a linear SVM classifier.
    """
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
    train_X_tfidf = vectorizer.fit_transform(train_X)
    val_X_tfidf, test_X_tfidf = map(vectorizer.transform, [val_X, test_X])
    test_predictions = []
    val_f1s = []
    cs = [0.1, 0.5, 1]

    for c in cs:
        logging.info(f'SVM with c={c}')
        classifier = svm.SVC(kernel='linear', C=c)
        classifier.fit(train_X_tfidf, train_y)
        val_predictions = classifier.predict(val_X_tfidf)
        val_f1s.append(precision_recall_fscore_support(val_y, val_predictions, average='weighted')[2])
        test_predictions.append(classifier.predict(test_X_tfidf))

    best_index = np.argmax(val_f1s)
    test_predictions = test_predictions[best_index]
    p, r, f, _ = precision_recall_fscore_support(test_y, test_predictions, average='weighted')
    return test_predictions, p, r, f, {"c": cs[best_index]}


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class FineTuneBert(object):
    def __init__(self,
                 pretrained_bert,
                 device,
                 train_X, train_y, val_X, val_y, test_X, test_y,
                 start_token="[CLS]",
                 end_token="[SEP]",
                 padding_idx=0,
                 max_seq_length=150):
        self.start_token = start_token
        self.end_token = end_token
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)
        self.num_labels = len(set(train_y))
        self.model = BertForSequenceClassification.from_pretrained(pretrained_bert, num_labels=self.num_labels).to(device)
        self.device = torch.device(device)
        self.label_map = None
        self.train, self.val, self.test = self.generate_features(train_X, train_y, val_X, val_y, test_X, test_y)

    def example_to_feature(self, text, label):
        tokens = self.tokenizer.tokenize(text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]

        tokens = [self.start_token] + tokens + [self.end_token]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Pad up to the sequence length.
        input_mask += [0] * (self.max_seq_length - len(input_ids))
        padding = [self.padding_idx] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = self.label_map[label]

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=label_id)

    def generate_features(self, train_X, train_y, val_X, val_y, test_X, test_y):
        """
        Initialize the BERT features once
        :return:
        """
        device = self.device
        logging.info('Generating train features...')

        if self.label_map is None:
            self.label_map = {label: i for i, label in enumerate(list(set(test_y)))}
            self.inversed_label_map = {i: label for label, i in self.label_map.items()}

        train_features = [self.example_to_feature(text, label) for text, label in tqdm.tqdm(zip(train_X, train_y))]
        train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long, device=device)
        train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long, device=device)
        train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long, device=device)
        train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long, device=device)

        logging.info('Generating val features...')
        val_features = [self.example_to_feature(text, label) for text, label in tqdm.tqdm(zip(val_X, val_y))]
        val_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long, device=device)
        val_input_mask = torch.tensor([f.input_mask for f in val_features], dtype=torch.long, device=device)
        val_segment_ids = torch.tensor([f.segment_ids for f in val_features], dtype=torch.long, device=device)
        val_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long, device=device)

        logging.info('Generating test features...')
        test_features = [self.example_to_feature(text, label) for text, label in tqdm.tqdm(zip(test_X, test_y))]
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long, device=device)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long, device=device)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long, device=device)
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long, device=device)

        return (train_input_ids, train_input_mask, train_segment_ids, train_label_ids), \
               (val_input_ids, val_input_mask, val_segment_ids, val_label_ids), \
               (test_input_ids, test_input_mask, test_segment_ids, test_label_ids)

    def fine_tune(self):
        """
        Fine-tune BERT to predict the name.
        """
        batch_sizes = [4]
        num_epochs = [4, 10]
        lrs = [1e-5, 1e-3]
        val_results, test_results, test_predictions = [], [], []
        setups = [{'batch_size': bs, 'num_epochs': ne, 'lr': lr}
                  for bs, ne, lr in itertools.product(batch_sizes, num_epochs, lrs)]

        train_input_ids, train_input_mask, train_segment_ids, train_label_ids = self.train
        val_input_ids, val_input_mask, val_segment_ids, val_label_ids = self.val
        test_input_ids, test_input_mask, test_segment_ids, test_label_ids = self.test

        for setup in setups:
            logging.info(setup)
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=setup['lr'])

            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0

            logging.info("***** Running training *****")
            logging.info("  Num examples = %d", len(train_input_ids))
            logging.info("  Batch size = %d", setup['batch_size'])

            train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=setup['batch_size'])

            for _ in tqdm.trange(setup['num_epochs'], desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Iteration")):
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = self.model(input_ids, segment_ids, input_mask, labels=label_ids.view(-1))[0]
                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    print(f"\r{(tr_loss / nb_tr_steps):.3f}", end='')

            logging.info('Validation')
            val_data = TensorDataset(val_input_ids, val_input_mask, val_segment_ids, val_label_ids)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=setup['batch_size'])

            self.model.eval()
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm.tqdm(val_dataloader, desc="Evaluating"):
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_mask, labels=None)[0]
                preds.extend(logits.detach().cpu().numpy().tolist())

            val_predictions = [np.argmax(pred) for pred in preds]
            val_y = val_label_ids.detach().cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(val_y, val_predictions, average='weighted')
            val_results.append(f)
            logging.info(f'Batch size: {setup["batch_size"]}, LR: {setup["lr"]}, Epochs: {setup["num_epochs"]}')
            logging.info(f'Validation Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}')

            logging.info('Test')
            test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=setup['batch_size'])

            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm.tqdm(test_dataloader, desc="Evaluating"):
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_mask, labels=None)[0]
                preds.extend(logits.detach().cpu().numpy().tolist())

            curr_test_predictions = [np.argmax(pred) for pred in preds]
            test_y = test_label_ids.detach().cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(test_y, curr_test_predictions, average='weighted')
            test_results.append((p, r, f))
            test_predictions.append([self.inversed_label_map[l] for l in curr_test_predictions])

        best_index = np.argmax(val_results)
        setup = setups[best_index]
        p, r, f = test_results[best_index]
        logging.info(f'Batch size: {setup["batch_size"]}, LR: {setup["lr"]}, Epochs: {setup["num_epochs"]}')
        logging.info(f'Test Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}')
        return test_predictions[best_index], p, r, f, setup


if __name__ == "__main__":
    main()
