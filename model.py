import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        '''
        Args:
        d_model: int, dimension of the word embedding
        vocab_size: int, number of words in the vocabulary
        ''' 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layer: A dictionary, produces a lookup that maps an index to an embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # as specified by paper: * math.sqrt(self.d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        '''
        Args:
        d_model: int, dimension of the position embedding (needs to be same as
        the input embeddings' dimensions)
        seq_len: int, maximum length of the sequence (sentence in this case)
        dropout: float, probability to apply dropout to prevent model from overfitting
        '''
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Build a matrix of shape (seq_len, d_model)
        # Need vectors of dimension d_model for each word in the sequence, for every word in the sequence (seq_len)
        # NOTE: Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # NOTE: Create a vector of shape (seq_len, 1)
        # .arange(...) -> tensor([0, 1, 2, ..., seq_len - 1 ])
        # .unsqueeze(1) -> tensor([[0],
        #                          ...
        #                         [seq_len - 1]])
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # numerator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator

        # Apply sin to even position
        pe[:,0::2] = torch.sin(position * div_term)
        # Apply cos to odd position
        pe[:,1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model)

        # To save tensor along with state of model when this python file is saved
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensures that self.pe remains constant and is not trained
        # Along the seq_len axis, take up to the sequence length only
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        
        super().__init__()
        # for each item in the batch, x_j_bar = (x_j - mu_j) / sqrt((sigma_j**2) + eps)
        # eps to prevent division by zero when denominator tends to 0
        self.eps = eps
        # nn.Parameter makes it a parameter that is tuned by the network
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        # Perform mean on the last dimension and prevent loss of dimension when mean is taken
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:

        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1 # bias = True by default
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) -- (linear 1) --> (batch, seq_len, d_ff) -- (linear 2) --> (batch, seq_len, d_model)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # Ensure that dimension of vector handlded by each head is an int
        assert d_model % h == 0, "d_model is not divisible by n"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v

        self.w_o = nn.Linear(d_model, d_model) # W_o
        self.dropout = nn.Dropout(dropout)

    # Call it with the class name rather than using an instance of the class
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # .transpose(-2, -1) --> transpose the last two dimensions
        # key.transpose(-2, -1) --> (batch, h, seq_len, d_k) --> (batch, h, d_k, seq_len)
        # query @ key.transpose(-2, -1) --> (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len ,seq_len)
        # QK^T
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Mask at indices where value of mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        # softmax(QK^T)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len ,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # attention_scores is returned as well for visualisation
        # softmax(QK^T)V
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # mask is to prevent interactions between specific pairs of tokens (deliberate) or when we want to prevent
        # interaction with filler words to meet the seq_len
        # to apply mask, we assign the results of matrix multiplcation between Q' and K'
        # --> small value of input to the softmax returns 0 essentially meaning 0 attention between that pair of words
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Split the embedding, not the sentence
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concat head outputs
        # .contiguous() simply puts the entire tensor in a continuous memory block
        # (batch, h, seq_len, d_k) -(x.transpose(1,2))-> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Matmul with W_o
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # x is the input that skipped the current layer while self.dropout(sublayer(self.norm(x)))
        # is the processed output of that skipped layer
        return x + self.dropout(sublayer(self.norm(x)))
    
# Create encoder block
class EncoderBlock(nn.Module):

    def __init__(self, attention_block: MultiHeadAttentionBlock,
                feed_forward_block: FeedForwardBlock,
                dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self.self_attention_block
        self.feed_forward_block = feed_forward_block
        # nn.ModuleList: Holds submodules (custom layers) in a list
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask: mask to apply to input of encoder since want to hide interaction of padding words with other words
        # in the encoder, we are trying to find out the relationship between words in the same sentence
        # hence, we are applying attention to words within the same sentence
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    # for multi-head attention (the middle block in the decoder block),
    # attention is calculated using query from the decoder while key and value from the encoder
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    # x: input of the decoder
    # encoder_output: output of the encoder
    # src_mask: mask applied to encoder
    # tgt_mask: mask applied to decoder
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # self attention within decoder's input
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # cross attention between encoder's output and decoder's input
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
# Projection layer: Project from embedding subspace to vocabulary subspace
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # Keep encoding, decoding and projecting as seperate functions for the following reasons:
    # 1. During inference, can reuse output of encoder
    # 2. Keep output seperate for visualising attention

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create fthe decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                              src_pos, tgt_pos, projection_layer)
    
    # Initialise the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer