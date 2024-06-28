import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

# imports from hugging face
from datasets import load_dataset
from tokenizers import Tokenizer 
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # Trains the tokenizer that creates the vocabulary given the list of sentences 
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'[lang]]

def get_or_build_tokenizer(config, ds, lang):
    # args: config of tokenizer, dataset, language for which tokenizer is built for
    
    # e.g., config['tokenizer_file'] = '../tokenizers/tokenizer_{lang}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        # special_tokens: list of tokens to be considered as special tokens
        # min_frequency: number of times a word need to be appear for it to be in the vocabulary
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split = 'train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])