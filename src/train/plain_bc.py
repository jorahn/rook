import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import Trainer, TrainingArguments
import torch
from datasets import load_dataset

from src.tokenizer.tokenizer import make_tokenizer
from src.model.model import make_model, make_config

tokenizer = make_tokenizer("src/tokenizer/rookworld_vocab.json")

config = make_config(
    vocab_size=2048,             # 32 should suffice for FEN encoding + 1968 actions for av predictor
                                # padding to the next power of 2 -> 2048
    pad_token_id=0,
    hidden_size=256,             # embedding dimension from the paper
    intermediate_size=1024,      # not specified
    num_hidden_layers=8,         # as in the paper
    num_attention_heads=8,       # as in the paper
    max_position_embeddings=78,  # as in the paper (for bc and sv predictors, +1 for av)
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    #device_map="auto",
)
model = make_model(config)

dataset = load_dataset("jrahn/rookworld_40m")

training_args = TrainingArguments(
    # 2 devices
    per_device_train_batch_size=512,  # bs 1024 in the paper on 4x 95G tpu, try to fit as much as possible ...
    gradient_accumulation_steps=1,    # ... else increase this
    gradient_checkpointing=False,     # save memory if needed, reduces speed
    bf16=True,
    learning_rate=4e-4,               # as in the paper
    #optim="adamw_torch_fused",
    #torch_compile=True,
    output_dir="tmp",
    per_device_eval_batch_size=256,
    eval_strategy="steps",
    eval_steps=1000,
    #num_train_epochs=3.0,            # 2.7-3.2 in the paper for ablations, 5.4 for full training
    max_steps=5e6,                    # 5e6 in the paper, 40m samples, bs 1024 -> 128 Epochs !?!
    lr_scheduler_type="cosine",
    warmup_steps=500,
    save_strategy="epoch",
    log_level="error",
    report_to="none",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    #compute_metrics=compute_metrics, # TODO
    optimizers=(None, None), # TODO: non-standard optimizer and scheduler
)

trainer.train()
