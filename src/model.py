import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import LlamaForSequenceClassification, LlamaConfig
from transformers import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from src.const import ACTION_SPACE, VOCAB

class RookTokenizer(PreTrainedTokenizerFast):
    # TODO: make it easier to use checkpoints from the hub
    # https://huggingface.co/docs/transformers/custom_models#sending-the-code-to-the-hub
    def __call__(self, *args, **kwargs):
        kwargs["return_token_type_ids"] = False
        return super().__call__(*args, **kwargs)

def make_model(config_dict):
    if config_dict["finetuning_task"] == "text-classification":
        return make_model_clf(config_dict)
    elif config_dict["finetuning_task"] == "text-generation":
        return make_model_lm(config_dict)
    else:
        raise ValueError(f"Unknown config finetuning_task: {config_dict['finetuning_task']}")

def make_model_clf(config_dict):
    # pad to multiple of 128
    config_dict["vocab_size"] = ((len(VOCAB) + 127) // 128) * 128
    config = LlamaConfig(**config_dict)
    label_to_id = {v: i for i, v in enumerate(ACTION_SPACE)}
    config.num_labels = len(ACTION_SPACE)
    config.label2id = label_to_id
    config.id2label = {id: label for label, id in label_to_id.items()}
    model = LlamaForSequenceClassification(config=config)
    return model

def make_model_lm(config_dict):
    # pad to multiple of 128
    config_dict["vocab_size"] = ((len(VOCAB) + len(ACTION_SPACE) + 3 + 127) // 128) * 128
    config = LlamaConfig(**config_dict)
    model = LlamaForCausalLM(config=config)
    return model


def make_tokenizer(task="clf", model_max_length=78):
    if task == "clf":
        return make_tokenizer_clf(model_max_length=model_max_length)
    elif task == "lm":
        return make_tokenizer_lm(model_max_length=model_max_length)
    else:
        raise ValueError(f"Unknown task: {task}")
    
def make_tokenizer_clf(model_max_length=78):
    vocab = VOCAB

    tokenizer = Tokenizer(BPE(
        vocab=dict(zip(vocab, range(len(vocab)))), 
        merges=[])
    )

    fast_tokenizer = RookTokenizer(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=False
    )
    return fast_tokenizer

def make_tokenizer_lm(model_max_length=192): # TODO verify lm max length
    vocab = VOCAB + ACTION_SPACE
    vocab += ["[OPTIONS]", "[VALUES]", "[ACTION]"]

    tokenizer = Tokenizer(BPE(
        vocab=dict(zip(vocab, range(len(vocab)))), 
        merges=[])
    )

    fast_tokenizer = RookTokenizer(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=False
    )
    return fast_tokenizer