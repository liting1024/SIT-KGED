from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
import os

from inference import run as inference
# from ..inference import inference

class AdapterSaveCallback(TrainerCallback):
    def __init__(self, LLM_NAME=None, LLM_PATH=None, VAL_PATH=None, TEST_PATH=None):
        super().__init__()
        self.LLM_NAME = LLM_NAME
        self.LLM_PATH = LLM_PATH
        self.VAL_PATH = VAL_PATH
        self.TEST_PATH = TEST_PATH

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        peft_model_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        model.llama_model.save_pretrained(peft_model_path) 
        torch.save(model.embeddings, os.path.join(peft_model_path, "embeddings.pth"))

        
        # if state.epoch > 1:
        ''' enable evaluation after each epoch '''
        #     print(f"Epoch {state.epoch}; Adapter saved to {peft_model_path}")
        #     inference(
        #         LLM_MODEL=self.LLM_NAME,
        #         LLM_PATH=self.LLM_PATH,
        #         lora_weights=peft_model_path,
        #         test_data_path=self.VAL_PATH,
        #     )
        #     inference(
        #         LLM_MODEL=self.LLM_NAME,
        #         LLM_PATH=self.LLM_PATH,
        #         lora_weights=peft_model_path,
        #         test_data_path=self.TEST_PATH,
        #     )
            