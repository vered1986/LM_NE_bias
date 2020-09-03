#!/bin/bash

device=$1

declare -a model_names=(gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased transfo-xl-wt103)
declare -a params=("--num_texts 50 --max_length 75 --p 0.9 --out_dir is_a_endings_p0.9_l75"
                   "--num_texts 50 --max_length 300 --p 0.9 --out_dir is_a_endings_p0.9_l300"
                   "--num_texts 50 --max_length 150 --k 25 --out_dir is_a_endings_k25_l150")
declare -a in_dirs=("is_a_endings_p0.9_l75" "is_a_endings_p0.9_l300" "is_a_endings_k25_l150")



# Generate all the endings
for param in "${params[@]}"
do
    python generate_texts.py --device ${device} ${param};
done


for model_name in "${model_names[@]}"
do
    for in_dir in "${in_dirs[@]}"
    do
        python predict_given_name.py --model_name ${model_name} --device ${device} --input_dir ${in_dir};
    done
done
