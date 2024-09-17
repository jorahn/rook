import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import LlamaForSequenceClassification, LlamaConfig
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from src.const import ACTION_SPACE, VOCAB

class CustomTokenizer(PreTrainedTokenizerFast):
    def __call__(self, *args, **kwargs):
        kwargs["return_token_type_ids"] = False
        return super().__call__(*args, **kwargs)

def make_model(config_dict):
    config = LlamaConfig(**config_dict)
    label_to_id = {v: i for i, v in enumerate(ACTION_SPACE)}
    config.num_labels = len(ACTION_SPACE)
    config.label2id = label_to_id
    config.id2label = {id: label for label, id in label_to_id.items()}
    model = LlamaForSequenceClassification(config=config)
    return model

def make_tokenizer(model_max_length=78):
    vocab = VOCAB

    tokenizer = Tokenizer(BPE(
        vocab=dict(zip(vocab, range(len(vocab)))), 
        merges=[])
    )

    fast_tokenizer = CustomTokenizer(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        clean_up_tokenization_spaces=False
    )
    return fast_tokenizer
