import os
import logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

def process_func(example):
    MAX_LENGTH = 1024
    system_prompt = "你是一个优秀的童话故事作家，请根据用户的要求创作一个完整、有教育意义的童话故事。"
    instruction_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    response_text = f"{example['output']}"
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

model_path = 'ORIGINAL_MODEL_PATH'
dataset_path = 'YOUR_DATASET_PATH(JSON_FILE_FORMAT)'

# loading tokenizers
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False, 
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # all linear
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.enable_input_require_grads()
model.print_trainable_parameters()

ds = load_dataset("json", data_files=dataset_path, split="train")
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="./output/Qwen_story_lora_100",
    per_device_train_batch_size=2,   
    gradient_accumulation_steps=8,      
    logging_steps=5,                    
    num_train_epochs=5,                 
    save_strategy="epoch",              
    learning_rate=2e-4,
    gradient_checkpointing=True,
    bf16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train() # training the model from scratch

final_output_dir = "MODEL_SAFETENSOR_SAVE_PATHS"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"训练完成！最终模型已保存至 {final_output_dir}")