# WORK IN PROGRESS

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import LlamaForSequenceClassification, LlamaConfig

def make_config(**kwargs):
    return LlamaConfig(**kwargs)

def make_model(config):
    return LlamaForSequenceClassification(config=config)

