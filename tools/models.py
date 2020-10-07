import torch
from torch import nn
from transformers import BertConfig, BertLayer


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
                 num_layers, num_lstm_layers, bilstm,
                 embed_dropout, dropout,
                 num_embeddings, embed_dim,
                 out_dim, num_features,
                 num_trans_layers, num_trans_attention_heads):
        super(guchioGRU1, self).__init__()
        assert embed_dim % 2 == 0
        hidden_dim = embed_dim * 3 + num_features
        # self.pred_len = pred_len
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        if num_lstm_layers > 0:
            if bilstm:
                self.lstm = nn.LSTM(
                    input_size=hidden_dim*2,
                    hidden_size=hidden_dim,
                    num_layers=num_lstm_layers,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True,
                )
                self.flstm = None
                self.blstm = None
            else:
                self.lstm = None
                self.flstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_lstm_layers,
                    dropout=dropout,
                    bidirectional=False,
                    batch_first=True,
                )
                self.blstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_lstm_layers,
                    dropout=dropout,
                    bidirectional=False,
                    batch_first=True,
                )
        else:
            self.lstm = None
            self.flstm = None
            self.blstm = None
        # self.position_embedding = nn.Embedding(
        #     num_embeddings=,
        #     embedding_dim=hidden_dim*2)
        bert_layers = []
        self.bert_config = BertConfig()
        self.bert_config.hidden_size = hidden_dim * 2
        self.bert_config.num_attention_heads = num_trans_attention_heads
        for _ in range(num_trans_layers):
            bert_layer = BertLayer(self.bert_config)
            bert_layers.append(bert_layer)
        self.bert_layers = nn.Sequential(*bert_layers)
        self.linear = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self,
                encoded_sequence,
                encoded_structure,
                encoded_predicted_loop_type,
                features):
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
        features = [feature.reshape(feature.shape[0], feature.shape[1], 1)
                    for feature in features]
        embed = torch.cat([embed_sequence, embed_structure,
                           embed_predicted_loop_type] + features, dim=-1)
        embed = self.embed_dropout(embed)
        output, hidden = self.gru(embed)
        if self.lstm:
            output, hidden = self.lstm(output)
        elif self.flstm:
            foutput = output[:, :, :output.shape[-1] // 2]
            boutput = torch.flip(output[:, :, output.shape[-1] // 2:], [1])
            # if self.fb_reverse:
            #     temp = foutput
            #     foutput = boutput
            foutput = self.flstm(foutput)[0]
            boutput = torch.flip(self.blstm(boutput)[0], [1])
            output = torch.cat([foutput, boutput], dim=-1)
        for bert_layer in self.bert_layers:
            output = bert_layer(output)[0]
        out = self.linear(output)
        # truncated = output[:, : self.pred_len, :]
        # out = self.linear(truncated)
        return out
