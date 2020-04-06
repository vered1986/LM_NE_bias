#!/bin/bash

cp ./data/all_names.tsv ./data/all_names_results.tsv

for M in openai-gpt,openai-gpt gpt2,gpt2 gpt2-medium,gpt2-medium gpt2-large,gpt2-large gpt2-xl,gpt2-xl transfo-xl,transfo-xl-wt103 xlnet,xlnet-base-cased xlnet,xlnet-large-cased; do
  IFS=',' read MT MN <<< "${M}"
  for CT in none news history personal; do
    echo $MT $MN $CT
    python next_word_prob.py --model_type=$MT --model_name_or_path=$MN --length=20 --names_file=./data/all_names.tsv --next_word_prob --context_type=$CT > .temp.tsv
    paste ./data/all_names_results.tsv .temp.tsv > .temp2.tsv
    mv .temp2.tsv ./data/all_names_results.tsv
  done
done
rm .temp.tsv
sed -e "s///" all_names_results.tsv > .temp3.tsv
mv .temp3.tsv all_names_results.tsv