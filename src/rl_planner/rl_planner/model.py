import torch
import torch.nn as nn
import math


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(PolicyNet, self).__init__()

        # graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # pointer
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                             key_padding_mask=node_padding_mask,
                                             attn_mask=edge_mask)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                            current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                        enhanced_node_feature,
                                                        node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_policy(self, current_node_feature, enhanced_current_node_feature,
                      enhanced_node_feature, current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                  current_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        logp = self.pointer(current_state_feature, neighboring_feature, edge_padding_mask)
        logp = logp.squeeze(1)

        return logp

    # @torch.compile
    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)
        logp = self.output_policy(current_node_feature, enhanced_current_node_feature,
                                  enhanced_node_feature, current_edge, edge_padding_mask)

        return logp


class QNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(QNet, self).__init__()

        # graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                             key_padding_mask=node_padding_mask,
                                             attn_mask=edge_mask)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                            current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                        enhanced_node_feature,
                                                        node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = current_edge.size()[1]
        current_state_feature = torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index,
                                                                                node_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask)

        return q_values
