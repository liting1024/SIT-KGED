<div align="center">
<h1> 🧑‍🦼SIT-KGED </h1>
<h3> <b>S</b>imply <b>I</b>nject <b>T</b>opology into LLM for <b>K</b>nowledge <b>G</b>raph <b>E</b>rror <b>D</b>etection </h3>
</div>

This paper extends KGED to a four-class classification task to identify which element of a triple is incorrect. To address this fine-grained reasoning challenge, we precompute high-order common neighbors between the head and tail entities to obtain topological evidence while reducing computational complexity. Furthermore, we introduce a mixture-of-experts adapter that maps both structural embeddings and high-order topological information into the text embedding space. 

It can support knowledge verification and enhance the quality of automatic knowledge graph construction and GraphRAG systems. For more details, please refer to our paper[📄](https://www2026.thewebconf.org/calls/short-papers.html).

## Environment
We recommend using Python 3.8+. Higher versions should also be compatible. To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Preprocess Datasets
Pre-train knowledge graph embeddings for building datasets and evaluation.
处理前的数据集存在preprocess raw
```bash
sh scripts/preprocess.sh
```

## Fine-tune & Inference
Download [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), and fine-tune it using the command below:
```bash
python finetune.py \
  --MICRO_BATCH_SIZE 16 \
  --BATCH_SIZE 16 \
  --EPOCHS 3 \
  --LEARNING_RATE 3e-4 \
  --train_on_inputs True \
  --add_eos_token False \
  --CUTOFF_LEN 256 \
  --LORA_R 64 \
  --LORA_ALPHA 16 \
  --LORA_DROPOUT 0.05 \
  --VAL_SET_SIZE 0 \
  --EVAL_STEPS 20 \
  --TARGET_MODULES '["q_proj", "k_proj", "v_proj", "o_proj"]' \
  --DATA_PATH "data/WN18RR/train.json" \
  --VAL_PATH "data/WN18RR/val.json" \
  --TEST_PATH "data/WN18RR/test.json" \
  --LLM_NAME "Llama-3.1-8B" \
  --LLM_PATH "YOUR_LLM_PATH" \
  --ENT_PATH "preprocess/WN18RR/TransE-D512/entity_embeddings.pth" \
  --REL_PATH "preprocess/WN18RR/TransE-D512/relation_embeddings.pth" \
  --PROMPT_TEMPLATE "alpaca" \
  --MODEL_NAME "SIT" \
  --REPORT_TO "wandb" \
  --WANDB_RUN_NAME "SIT-KGED" \
  --OUTPUT_DIR "results/WN18RR/SIT" \
  > results/WN18RR/train.log &
python inference_kopa.py \
  --LLM_PATH "YOUR_LLM_PATH" \
  --LORA_PATH "YOUR_FineTune_LORA_SAVE_PATH" \
  --TEST_PATH "data/WN18RR/test.json" \
  --DATA_NAME "WN18RR" \
  > infer.log &
```

## Baselines



## Acknowledgements
We thank the following open-source projects for their contributions: [PyKEEN](https://github.com/pykeen/pykeen), [KoPA](https://github.com/zjukg/KoPA), [KG-LLM](https://github.com/yao8839836/kg-llm), [MPLP](https://github.com/Barcavin/efficient-node-labelling) 





