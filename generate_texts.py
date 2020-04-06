"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import os
import re
import tqdm
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import GPT2Tokenizer, XLNetTokenizer, TransfoXLTokenizer, OpenAIGPTTokenizer
from transformers import GPT2LMHeadModel, XLNetLMHeadModel, TransfoXLLMHeadModel, OpenAIGPTLMHeadModel

logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

logger = logging.getLogger(__name__)

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-medium': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-large': (GPT2LMHeadModel, GPT2Tokenizer),
    'gpt2-xl': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet-base-cased': (XLNetLMHeadModel, XLNetTokenizer),
    'xlnet-large-cased': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl-wt103': (TransfoXLLMHeadModel, TransfoXLTokenizer)
}


def main():
    """
    Generate num_texts number of texts of length max_length for each model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=1000, required=False,
                        help="Number of texts to generate for each model")
    parser.add_argument("--max_length", default=75, type=int, required=False, help="Maximum text length")
    parser.add_argument("--device", default='cuda:0', type=str, required=False, help="CPU/GPU device")
    parser.add_argument("--prefix", default='The ', type=str, required=False, help="The beginning of the text")
    parser.add_argument("--out_dir", default='texts', type=str, required=False, help="Output directory")
    parser.add_argument("--p", default=0.5, type=float, required=False, help="p for nucleus sampling")
    parser.add_argument("--batch_size", default=20, type=int, required=False, help="How many texts to generate at once")
    args = parser.parse_args()
    logger.debug(args)

    logger.debug(f"Initializing {args.device}")
    device = torch.device(args.device)

    for model_name in MODEL_CLASSES.keys():
        model, tokenizer = init_model(model_name, device)
        curr_dir = f'{args.out_dir}/{model_name}/'

        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        with tqdm.tqdm(total=args.num_texts) as pbar:
            for indices in chunks(list(range(args.num_texts)), args.batch_size):
                texts = generate_ending(
                    model_name, model, tokenizer, args.prefix, args.device, top_p=args.p, 
                    num_samples=args.batch_size, length=args.max_length)

                for i, text in zip(indices, texts):
                    pbar.update(1)
                    logger.info(text)
                    with open(f'{curr_dir}/{i}.txt', 'w') as f_out:
                        f_out.write(text)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def top_p_filtering(logits: torch.Tensor,
                    top_p: float = 0.0,
                    filter_value: float = -float('Inf')):
    """
    Filter a distribution of logits using nucleus (top-p) filtering

    logits: logits distribution shape (vocabulary size)
    top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(model: PreTrainedModel,
                    length: int,
                    context: int,
                    num_samples: int = 1,
                    temperature: float = 1.0,
                    top_p: float = 0.0,
                    is_xlnet: bool = False,
                    device='cpu'):
    """
    Sample a sequence of length tokens
    :param model: a pre-trained LM
    :param length: maximum text length
    :param context: the prefix on which the generation is conditioned
    :param num_samples: number of texts to generated
    :param temperature: default = 1
    :param top_p: p for nucleus sampling
    :param is_xlnet: special treatment for XLNet
    :param device: CUDA / CPU device
    :return: the generated texts
    """
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked),
                # attention mask and target mapping (see model docstring)
                dummy = torch.zeros((num_samples, 1), dtype=torch.long, device=device)
                input_ids = torch.cat((generated, dummy), dim=1)
                seq_len = input_ids.shape[1]
                perm_mask = torch.zeros((num_samples, seq_len, seq_len), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((num_samples, 1, seq_len), dtype=torch.float, device=device)
                target_mapping[:, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / temperature
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return generated


def init_model(model_name: str,
               device: str):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    model_class, tokenizer_class = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_ending(model_name: str,
                    model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    beginning: str,
                    device: str,
                    top_p: float = 0.0,
                    temperature: float = 1.0,
                    length: int = 75,
                    num_samples: int = 1):
    """
    Generate an ending for the beginning of the text
    :param model_name: from MODEL_CLASSES
    :param model: the pre-trained LM
    :param tokenizer: the pre-trained tokenizer
    :param beginning: text on which the generation is conditioned
    :param device: CUDA / CPU device
    :param top_p: p for nucleus sampling
    :param temperature: default = 1
    :param length: the maximum length to sample
    :param num_samples: how many texts to generate at once
    :return: the text
    """
    prefix = beginning
    if "transfo-xl" in model_name or "xlnet" in model_name:
        prefix = " ".join((PADDING_TEXT, beginning))

    context_tokens = tokenizer.encode(prefix)
    out = sample_sequence(
        model=model, context=context_tokens, length=length, temperature=temperature,
        top_p=top_p, is_xlnet="xlnet" in model_name, device=device, num_samples=num_samples)
    out = out[:, len(context_tokens):].tolist()
    texts = [tokenizer.decode(t, clean_up_tokenization_spaces=True, skip_special_tokens=True) for t in out]
    texts = [re.sub(' +', ' ', ' '.join((beginning, text))) for text in texts]
    return texts


if __name__ == '__main__':
    main()


