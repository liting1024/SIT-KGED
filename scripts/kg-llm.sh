# !/usr/bin/env bash


DATASET="WN18RR"
EPOCHS=3
# WN18RR, fb15k237, codexmedium: 3; codexsmall: 8

CUDA_VISIBLE_DEVICES=6 python baseline/kg-llm/lora_fintune.py \
  --MICRO_BATCH_SIZE 70 \
  --BATCH_SIZE 70 \
  --EPOCHS $EPOCHS \
  --LEARNING_RATE 3e-4 \
  --CUTOFF_LEN 50 \
  --LORA_R 64 \
  --LORA_ALPHA 16 \
  --LORA_DROPOUT 0.05 \
  --TARGET_MODULES '["q_proj", "v_proj", "k_proj", "o_proj"]' \
  --DATA_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/train_4c.json" \
  --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/models/Llama3-8B-$DATASET/R=64" 

DATASET="WN18RR"
CUDA_VISIBLE_DEVICES=2 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/models/Llama3-8B-$DATASET/R=8/checkpoint-3722" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=8/e1_test.csv" &
CUDA_VISIBLE_DEVICES=2 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/models/Llama3-8B-$DATASET/R=64/checkpoint-7444" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=64/e2_test.csv" &
CUDA_VISIBLE_DEVICES=3 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/models/Llama3-8B-$DATASET/R=64/checkpoint-11166" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=64/e3_test.csv" &

nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=8/e1_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=8/e1_test_eval.txt


DATASET="codexsmall"
CUDA_VISIBLE_DEVICES=0 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/models/Llama3-8B-$DATASET/R=64/checkpoint-8460" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=64/e1_test.csv" &
nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=64/e1_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/baseline/kg-llm/results/Llama3-8B-$DATASET/R=64/e1_test_eval.txt

# 换回自己的代码
CUDA_VISIBLE_DEVICES=3 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  20     --BATCH_SIZE  20     --EPOCHS  1      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 8     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/WN18RR/TransE-D512/train_4c.json"     --VAL_PATH "data/WN18RR/TransE-D512/val_4c.json"     --TEST_PATH "data/WN18RR/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/WN18RR/TransE-D512/entity_embeddings.pth"     --REL_PATH "data/WN18RR/TransE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "WN18RR"  > /home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &

CUDA_VISIBLE_DEVICES=4 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  20     --BATCH_SIZE  20     --EPOCHS  1      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 8     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/WN18RR/TransE-D512/train_4c.json"     --VAL_PATH "data/WN18RR/TransE-D512/val_4c.json"     --TEST_PATH "data/WN18RR/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/WN18RR/TransE-D512/entity_embeddings.pth"     --REL_PATH "data/WN18RR/TransE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "WN18RR"  > /home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &

CUDA_VISIBLE_DEVICES=5 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  20     --BATCH_SIZE  20     --EPOCHS  1      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 8     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/WN18RR/TransE-D512/train_4c.json"     --VAL_PATH "data/WN18RR/TransE-D512/val_4c.json"     --TEST_PATH "data/WN18RR/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/WN18RR/TransE-D512/entity_embeddings.pth"     --REL_PATH "data/WN18RR/TransE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "WN18RR"  > /home/liting/Projects/KoPA-HoA/WN18RR/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &

CUDA_VISIBLE_DEVICES=3 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  24     --BATCH_SIZE  24     --EPOCHS  7      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 64     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/codexsmall/TransE-D512/train_4c.json"     --VAL_PATH "data/codexsmall/TransE-D512/val_4c.json"     --TEST_PATH "data/codexsmall/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/codexsmall/RotatE-D512/entity_embeddings.pth"     --REL_PATH "data/codexsmall/RotatE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/codexsmall/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "codexsmall"  > /home/liting/Projects/KoPA-HoA/codexsmall/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &

DATASET="WN18RR"
CUDA_VISIBLE_DEVICES=5 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/checkpoint-13026" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test.csv" &
nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test_eval.txt
CUDA_VISIBLE_DEVICES=5 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/checkpoint-26052" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e2_test.csv" 
nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e2_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e2_test_eval.txt
CUDA_VISIBLE_DEVICES=5 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/checkpoint-39078" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e3_test.csv" 
nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e3_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e3_test_eval.txt

DATASET="codexsmall"
CUDA_VISIBLE_DEVICES=3 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  20     --BATCH_SIZE  20     --EPOCHS  7      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 8     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/codexsmall/TransE-D512/train_4c.json"     --VAL_PATH "data/codexsmall/TransE-D512/val_4c.json"     --TEST_PATH "data/codexsmall/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/codexsmall/RotatE-D512/entity_embeddings.pth"     --REL_PATH "data/codexsmall/RotatE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/codexsmall/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "codexsmall"  > /home/liting/Projects/KoPA-HoA/codexsmall/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &
    
CUDA_VISIBLE_DEVICES=4 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/checkpoint-4934" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test.csv" &
nohup python baseline/kg-llm/eval.py \
  --PRED_PATH "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test.csv" \
  --TEST_PATH "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" > /home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test_eval.txt

DATASET="fb15k237"
CUDA_VISIBLE_DEVICES=6 python baseline/kg-llm/lora_infer.py \
  --BASE_MODEL "/home/HF_Model/meta-llama/Llama-3.1-8B" \
  --LORA_WEIGHTS "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/checkpoint-34015" \
  --finput "/home/liting/Projects/KoPA-HoA/data/$DATASET/TransE-D512/test_4c.json" \
  --foutput "/home/liting/Projects/KoPA-HoA/$DATASET/Llama-3.1-8B/V0/4c_KG-LLM/e1_test.csv" &

DATASET="codexmedium"
CUDA_VISIBLE_DEVICES=1 nohup python finetune-kg-llm.py     --MICRO_BATCH_SIZE  20     --BATCH_SIZE  20     --EPOCHS  3      --LEARNING_RATE 0.0003     --train_on_inputs True     --add_eos_token False     --CUTOFF_LEN 256     --LORA_R 8     --LORA_ALPHA 16     --LORA_DROPOUT 0.05     --VAL_SET_SIZE 0     --EVAL_STEPS 20     --TARGET_MODULES "['q_proj', 'v_proj', 'k_proj', 'o_proj']"     --TASK "4c"     --DATA_PATH "data/codexmedium/TransE-D512/train_4c.json"     --VAL_PATH "data/codexmedium/TransE-D512/val_4c.json"     --TEST_PATH "data/codexmedium/TransE-D512/test_4c.json"     --LLM_NAME "Llama-3.1-8B"     --LLM_PATH "/home/HF_Model/meta-llama/Llama-3.1-8B"     --HOA_PATH "data/codexmedium/TransE-D512/entity_embeddings.pth"     --REL_PATH "data/codexmedium/TransE-D512/relation_embeddings.pth"     --PROMPT_TEMPLATE "alpaca"     --MODEL_NAME "KoPAWithHoAV0"     --REPORT_TO "wandb"     --WANDB_RUN_NAME "4c_KG-LLM"     --OUTPUT_DIR "/home/liting/Projects/KoPA-HoA/codexmedium/Llama-3.1-8B/V0/4c_KG-LLM/"     --resume_from_checkpoint None     --DATASET "codexmedium"  > /home/liting/Projects/KoPA-HoA/codexmedium/Llama-3.1-8B/V0/4c_KG-LLM/log.txt &