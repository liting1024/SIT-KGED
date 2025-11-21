#!/bin/bash


datasets=("WN18RR" "fb15k237" "codexsmall" "codexmedium")
models=("TransE" "DistMult" "RGCN" "ConvE" "RotatE")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Training $model on $dataset"
        python preprocess/train_KGE.py \
            --DATASET "$dataset" \
            --MODEL "$model" \
            --DIM 512 

        if [ "$model" == "TransE" ]; then
            echo "Building dataset $dataset"
            python preprocess/build_datasets.py \
                --DATASET "$dataset" \
                --MODEL "${model}-D512" \
                --SENTENCE_PATH "YOUR_SENTENCE_EMBEDDING_PATH_HERE"
        fi
    done
done

wait
echo "All training jobs completed!"





