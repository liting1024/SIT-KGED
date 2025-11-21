import os
import fire

import json
import torch
import transformers
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import pandas as pd

transformers.logging.set_verbosity_error()  # ignore warnings only error


prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert in knowledge graph reasoning. Your task is to classify a given knowledge graph triple into one of the following categories: Triple Correct, Head Entity Error, Tail Entity Error, Relation Error.
### Input:
{}
### Response:

"""  # for alpaca


def load_test_dataset(path):
    test_dataset = json.load(open(path, "r", encoding="utf-8"))
    return test_dataset


def run(
    cuda="cuda:0",
    LLM_MODEL="Llama-3.1-8B",
    LLM_PATH="/home/HF_Model/meta-llama/Llama-3.1-8B",
    LORA_PATH="",
    TEST_PATH="data/WN18RR/test.json",
):
    print("Current Dir ", os.getcwd())

    embedding_path = f"{LORA_PATH}/embeddings.pth"

    test_dataset = load_test_dataset(TEST_PATH)
    kg_embeddings = (
        torch.load(embedding_path, weights_only=False).to(torch.float16).to(cuda)
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, torch_dtype=torch.float16
    ).to(cuda)

    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        torch_dtype=torch.float16,
    ).to(cuda)
    # unwind broken decapoda-research config

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model = model.eval()
    result = []
    for i, data in enumerate(test_dataset):
        ent = data["input"]
        ans = data["output"]
        ids = data["embedding_ids"]
        ids = torch.LongTensor(ids).reshape(1, -1).to(cuda)
        prefix = kg_embeddings(ids)
        prompt = prompt_template.format(ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)
        token_embeds = model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)
        generate_ids = model.generate(inputs_embeds=input_embeds, max_new_tokens=16)
        context = tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = response.replace(context, "").strip()
        # print(response + '\n')  # for debug
        result.append({"answer": ans, "predict": response})

    result_df = pd.DataFrame(result)
    result_df.to_csv(f"{LORA_PATH}/predict_result.csv", index=False)

    answer = []
    predict = []

    label_map = {
        "Triple Correct": 0,
        "Head Entity Error": 1,
        "Tail Entity Error": 2,
        "Relation Error": 3,
    }

    def match_label(text):
        for k, v in label_map.items():
            if k in text:
                return v
        return 0

    for data in result:
        answer.append(match_label(data["answer"]))
        predict.append(match_label(data["predict"]))

    averages = ["weighted", "macro"]
    for avg in averages:
        acc = accuracy_score(y_true=answer, y_pred=predict)
        p = precision_score(y_true=answer, y_pred=predict, average=avg)
        r = recall_score(y_true=answer, y_pred=predict, average=avg)
        f1 = f1_score(y_true=answer, y_pred=predict, average=avg)
        print(f"{avg}\t{acc:.4f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}")


if __name__ == "__main__":
    fire.Fire(run)
