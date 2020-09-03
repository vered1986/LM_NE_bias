import os
import re
import tqdm
import json
import torch
import logging
import argparse

from transformers import AutoModelWithLMHead, AutoTokenizer

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)


MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'openai-gpt',
          'xlnet-base-cased', 'xlnet-large-cased', 'transfo-xl-wt103']


def main():
    """
    Generate num_texts number of texts of length max_length for each model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=50, required=False,
                        help="Number of texts to generate for each model")
    parser.add_argument("--max_length", default=75, type=int, required=False, help="Maximum text length")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help="CPU/GPU device")
    parser.add_argument("--out_dir", default='texts', type=str, required=False, help="Output directory")
    parser.add_argument("--p", default=0, type=float, required=False, help="p for nucleus sampling")
    parser.add_argument("--k", default=0, type=int, required=False, help="k for top k sampling")
    parser.add_argument("--beams", default=0, type=int, required=False, help="Number of beams for beam search")
    parser.add_argument("--temperature", default=1.0, type=float, required=False, help="temperature for sampling")
    args = parser.parse_args()
    logger.debug(args)

    logger.debug(f"Initializing {args.device}")
    device = torch.device(args.device)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    people = [json.loads(line.strip()) for line in open("names.jsonl")]

    for model_name in MODELS:
        tokenizer, model = init_model(model_name, device)

        with open(os.path.join(args.out_dir, model_name) + ".jsonl", "w") as f_out:
            for person in tqdm.tqdm(people):
                prompt = f'{person["name"]} is a'
                texts = generate_ending(tokenizer, model, args, prompt, device)

                for text in texts:
                    logger.info(text)
                    text = re.sub(rf'\b{person["name"]}\b', "[NAME]", text, flags=re.IGNORECASE)
                    person_with_text = {k: v for k, v in person.items()}
                    person_with_text.update({"text": text})
                    f_out.write(json.dumps(person_with_text) + "\n")


def init_model(model_name: str, device, do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_ending(tokenizer, model, args, prompt, device):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_tokens, device=device).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=args.num_texts
    )

    preds = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return preds


if __name__ == '__main__':
    main()


