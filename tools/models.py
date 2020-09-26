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
    def __init__(self, seq_len=107, pred_len=68,
                 layer_num=3, dropout=0.5,
                 num_embeddings=14, embed_dim=128,
                 hidden_dim=128, hidden_layers=3, out_dim=3):
        super(guchioGRU1, self).__init__()
        assert embed_dim == hidden_dim
        self.pred_len = pred_len
        self.embeding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim)
        self.grus = []
        for _ in range(layer_num):
            self.grus.append(
                nn.GRU(
                    input_size=embed_dim * 3,
                    hidden_size=hidden_dim,
                    num_layers=hidden_layers,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True,
                )
            )
        self.linear = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self,
                encoded_sequence,
                encoded_structure,
                encoded_predicted_loop_type):
        seqs = torch.cat([encoded_sequence, encoded_structure,
                          encoded_predicted_loop_type], dim=1)
        embed = self.embeding(seqs)
        output = embed
        for gru in self.grus:
            output, hidden = self.gru(output)
        truncated = output[:, : self.pred_len, :]
        out = self.linear(truncated)
        return out
