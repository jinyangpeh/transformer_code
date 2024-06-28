import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    
    # To leverage Python's built-in behavior and make your class instances work seamlessly with functions like len(), you should define the __len__(self) method.
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenise and map each word into the corresponding ID
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Pad number of padding tokens needed for encoder and decoder
        # - 2 because we need to add 'SOS' and 'EOS' tokens to the sequence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # - 1 because we odd 'SOS' to decoder side and 'EOS' to label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure that sequence length chosen is long enough to represent all the sentences in our dataset
        # i.e. to say number of padding tokens should never be negative
        # too small seq_len we throw an exception
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
    
        # one sentence sent to input of encoder, another to input of decoder and one sentence is the one we expect as the output of the decoder (the ground truth, aka. the target/label)
        
        # NOTE: Encoder input - Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # NOTE: Decoder input - Add sos_token only
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # NOTE: Expected Decoder output (groundtruth/target/label) - Add eos_token only
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len,)
            "decoder_input": decoder_input, # (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) # (batch dimension, sequence dimension, sequence length)
            # Need a special mask since each word can only look at previous words and non-padding words
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  & causal_mask(decoder_input.size(0)) # (1,seq_len) & (1, seq_len, seq_len)
        }

# TODO: STOP AT 1:48:01
def causal_mask():
    raise NotImplementedError()