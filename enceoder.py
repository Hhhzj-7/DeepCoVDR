import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import nn
import torch.nn.functional as F
import copy
import math



# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

# word embedding and position encoding
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
# self attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads    # multi-heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, fusion):
        # for cross-attention module
        if fusion:
           
            # Q, K, V for drug in cross-attention
            mixed_query_layer = self.query(hidden_states[0])
            mixed_key_layer = self.key(hidden_states[0])
            mixed_value_layer = self.value(hidden_states[0])
            
            # Q, K, V for cell line in cross-attention
            mixed_query_layer1 = self.query2(hidden_states[1])
            mixed_key_layer1 = self.key2(hidden_states[1])
            mixed_value_layer1 = self.value2(hidden_states[1])

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            query_layer1 = self.transpose_for_scores(mixed_query_layer1)
            key_layer1 = self.transpose_for_scores(mixed_key_layer1)
            value_layer1 = self.transpose_for_scores(mixed_value_layer1)

            # attention scores for drug in cross-attention
            attention_scores = torch.matmul(query_layer, key_layer1.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + attention_mask

            # attention scores for cell line in cross-attention
            attention_scores1 = torch.matmul(query_layer1, key_layer.transpose(-1, -2))
            attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
            attention_scores1 = attention_scores1 + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

            attention_probs = self.dropout(attention_probs)
            attention_probs1 = self.dropout(attention_probs1)

            
            context_layer = torch.matmul(attention_probs1, value_layer)
            context_layer1 = torch.matmul(attention_probs, value_layer1)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()


            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        
            context_layer = context_layer.view(*new_context_layer_shape)
            context_layer1 = context_layer1.view(*new_context_layer_shape1)

            context_layer = torch.cat((context_layer.unsqueeze(0), context_layer1.unsqueeze(0)),0)
        # for graphtransformer
        else:
            # Q, K, V for drug in graphtransformer
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # attention scores for drug in graphtransformer
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            
        return context_layer
    
# output of self-attention
class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
# attention layer  
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask, fusion):
        self_output = self.self(input_tensor, attention_mask, fusion)
        if fusion:
            input_tensor = torch.cat((input_tensor[0].unsqueeze(0), input_tensor[1].unsqueeze(0)),0)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
# after attention    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
# output
class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Transformer encoder
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, fusion):
        attention_output = self.attention(hidden_states, attention_mask, fusion)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

# multi-heads transformer encoder
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, fusion,output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, fusion)
        return hidden_states
    
