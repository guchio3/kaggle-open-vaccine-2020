import torch
from torch import nn


class EMA(object):
    def __init__(self, model, mu, level='batch', n=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if True or param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if True or param.requires_grad:
                new_average = (1 - self.mu) * param.data + \
                    self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if True or param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level == 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level == 'epoch':
            self._update(model)


class guchioGRU1(nn.Module):
    def __init__(self,  # pred_len
                 num_layers, dropout,
                 num_embeddings, embed_dim,
                 out_dim):
        super(guchioGRU1, self).__init__()
        assert embed_dim % 2 == 0
        hidden_dim = embed_dim * 3
        # self.pred_len = pred_len
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self,
                encoded_sequence,
                encoded_structure,
                encoded_predicted_loop_type):
        embed_sequence = self.embedding(encoded_sequence)
        embed_structure = self.embedding(encoded_structure)
        embed_predicted_loop_type = self.embedding(encoded_predicted_loop_type)
        # seqs = torch.cat([encoded_sequence, encoded_structure,
        #                   encoded_predicted_loop_type], dim=1)
        # seqs = torch.stack(
        #     [encoded_sequence, encoded_structure, encoded_predicted_loop_type],
        #     dim=1)
        # embed = self.embedding(seqs)
        # output = embed
        embed = torch.cat([embed_sequence, embed_structure,
                           embed_predicted_loop_type], dim=-1)
        output, hidden = self.gru(embed)
        out = self.linear(output)
        # truncated = output[:, : self.pred_len, :]
        # out = self.linear(truncated)
        return out
