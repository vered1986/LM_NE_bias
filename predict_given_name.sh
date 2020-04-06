#!/bin/bash

device=$1

declare -a model_names=(gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased transfo-xl-wt103)

# Generate all the endings
python generate_texts.py --num_texts 50 --max_length 150 --device ${device} --p 0.9 --out_dir is_a_endings; 

for model_name in "${model_names[@]}"
do
    python predict_given_name.py --model_name ${model_name} --device ${device}; 
done
