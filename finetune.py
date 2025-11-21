import os
import fire

os.environ["WANDB_PROJECT"] = "KGED"
os.environ["WANDB_LOG_MODEL"] = "false"  # checkpoint dont upload
os.environ["WANDB_MODE"] = "offline"

import sys
import torch
import torch.nn as nn

from datasets import load_dataset
from utils.prompter import Prompter
import transformers

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    # prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from utils.peft_trainer import AdapterSaveCallback

import importlib


def dynamic_import(module_name, class_name):
    module = importlib.import_module("modules." + module_name)
    return getattr(module, class_name)


def run(
    DATA_PATH,
    VAL_PATH,
    TEST_PATH,
    ENT_PATH,
    REL_PATH,
    DATA_NAME,

    MICRO_BATCH_SIZE=8,  # this could actually be 5 but i like powers of 2
    BATCH_SIZE=8,
    EPOCHS=2,  # we don't always need 3 tbh
    LEARNING_RATE=3e-4,
    # Token
    train_on_inputs=True,  # if False, masks out inputs in loss
    add_eos_token=False,
    CUTOFF_LEN=256,  # 256 accounts for about 96% of the data
    # LORA Parameters
    LORA_R=64,
    LORA_ALPHA=16,
    LORA_DROPOUT=0.05,
    VAL_SET_SIZE=0,
    EVAL_STEPS=20,
    TARGET_MODULES=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    LLM_NAME="Llama-3.1-8B",
    LLM_PATH="/home/HF_Model/meta-llama/Llama-3.1-8B",
    PROMPT_TEMPLATE="llama",
    MODEL_NAME="SIT",
    REPORT_TO="none",
    WANDB_RUN_NAME="test",
    OUTPUT_DIR="fb15k237/llama-7b/test",
):
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    # device and ddp
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    model = (
        AutoModelForCausalLM.from_pretrained(
            LLM_PATH,
            # load_in_8bit=True
        )
        .half()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).cuda()

    rel_embeddings = torch.load(REL_PATH, weights_only=False).to(torch.float16).cuda()
    ent_embeddings = torch.load(ENT_PATH)
    if type(ent_embeddings) is list:
        ent_embeddings = [feat.to(torch.float16).cuda() for feat in ent_embeddings]
    else:
        ent_embeddings = ent_embeddings.to(torch.float16).cuda()

    our_model = dynamic_import(MODEL_NAME, MODEL_NAME)(
        model,
        num_prefix=1,
        ent_embeddings=ent_embeddings,
        rel_embeddings=rel_embeddings,
        dataset=DATA_NAME
    )

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    _, file_extension = os.path.splitext(DATA_PATH)
    data = load_dataset(path=file_extension[1:], data_files=DATA_PATH)

    prompter = Prompter(PROMPT_TEMPLATE)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if VAL_SET_SIZE == -1:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

        _, file_extension = os.path.splitext(VAL_PATH)
        val_data = load_dataset(path=file_extension[1:], data_files=VAL_PATH)
        val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt)
    elif VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    elif VAL_SET_SIZE == 0:
        train_data = (
            data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=16)
        )
        val_data = None

    trainer = transformers.Trainer(
        model=our_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=100,
            eval_strategy="steps" if VAL_SET_SIZE != 0 else "no",
            eval_steps=EVAL_STEPS if VAL_SET_SIZE != 0 else None,
            save_strategy="epoch",  # best or steps or epoch
            save_steps=200,
            output_dir=OUTPUT_DIR,
            save_total_limit=7,
            load_best_model_at_end=True if VAL_SET_SIZE != 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            optim="adamw_torch",
            report_to=REPORT_TO,
            run_name=WANDB_RUN_NAME,
            save_safetensors=False,
            disable_tqdm=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            AdapterSaveCallback(
                LLM_NAME=LLM_NAME,
                LLM_PATH=LLM_PATH,
                VAL_PATH=VAL_PATH,
                TEST_PATH=VAL_PATH,
            )
        ],
    )

    model.config.use_cache = False
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    torch.save(our_model.embeddings, f"{OUTPUT_DIR}/embeddings.pth")




if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current dir: ", current_dir)
    os.chdir(current_dir)
    fire.Fire(run)
