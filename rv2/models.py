"""Model definitions: BiLSTM+attention, MTL, and PyTorch ports of the Keras CNN models."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- BiLSTM
class BiLSTMAttentionClassifier(nn.Module):
    """Encoder -> 2-layer BiLSTM -> additive attention -> linear."""

    def __init__(self, transformer, hidden_size, lstm_hidden, num_labels, dropout=0.3, lstm_layers=2):
        super().__init__()
        self.transformer = transformer
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden, batch_first=True,
                              bidirectional=True, num_layers=lstm_layers,
                              dropout=dropout if lstm_layers > 1 else 0.0)
        self.attention_linear = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.attention_context = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask, **kw):
        h = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.bilstm(h)
        scores = self.attention_context(torch.tanh(self.attention_linear(lstm_out))).squeeze(-1)
        scores = scores + (1.0 - attention_mask.float()) * -1e9
        a = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(a.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(self.dropout(ctx))


# ---------------------------------------------------------------- MTL
class MTLModel(nn.Module):
    """Shared encoder, [CLS] -> dropout -> task head."""

    def __init__(self, encoder, hidden_size, num_hs, num_sent, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        def head(n):
            return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Dropout(dropout), nn.Linear(hidden_size, n))
        self.hs_head = head(num_hs)
        self.sent_head = head(num_sent)

    def forward(self, input_ids, attention_mask, task="hs", **kw):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.hs_head(cls) if task == "hs" else self.sent_head(cls)


# ---------------------------------------------------------------- CNN (Keras -> PyTorch port)
def _glorot_(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)          # Keras glorot_uniform default
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)     # Keras Embedding default


class KerasBN(nn.BatchNorm1d):
    """Keras BatchNormalization defaults: momentum 0.99 (torch 0.01), eps 1e-3."""

    def __init__(self, n):
        super().__init__(n, eps=1e-3, momentum=0.01)


class CNNBaseline(nn.Module):
    """CNN baseline: 4x Conv1D(same) [512,256,32,32] -> GlobalMaxPool ->
    Dense(512, L1L2 kernel, L2 bias, L2 activity) -> Dropout -> BN -> Dropout -> Dense(C)."""

    def __init__(self, vocab, num_classes, emb=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        chans = [emb, 512, 256, 32, 32]
        self.convs = nn.ModuleList([nn.Conv1d(chans[i], chans[i + 1], 3, padding=1) for i in range(4)])
        self.dense = nn.Linear(32, 512)
        self.drop1, self.bn, self.drop2 = nn.Dropout(0.5), KerasBN(512), nn.Dropout(0.5)
        self.out = nn.Linear(512, num_classes)
        self.apply(_glorot_)
        self._act = None

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(h))
        h = h.max(dim=2).values
        h = self.dense(h)
        self._act = h
        return self.out(self.drop2(self.bn(self.drop1(h))))

    def reg_loss(self):
        w, b = self.dense.weight, self.dense.bias
        l = 1e-5 * w.abs().sum() + 1e-4 * (w ** 2).sum() + 1e-4 * (b ** 2).sum()
        if self._act is not None:                      # Keras divides activity loss by batch size
            l = l + 1e-5 * (self._act ** 2).sum() / self._act.shape[0]
        return l


class SCMMMA(nn.Module):
    """SCM+MMA: 4x Conv1D(valid) [512,256,128,64] -> MMA(pool 2) ->
    Dense(32, relu) per step -> Dropout -> BN -> Dropout -> Flatten -> Dense(C)."""

    def __init__(self, vocab, num_classes, emb=128, max_len=80):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        chans = [emb, 512, 256, 128, 64]
        self.convs = nn.ModuleList([nn.Conv1d(chans[i], chans[i + 1], 3) for i in range(4)])
        steps = (max_len - 8) // 2
        self.dense = nn.Linear(64, 32)
        self.drop1, self.bn, self.drop2 = nn.Dropout(0.5), KerasBN(32), nn.Dropout(0.5)
        self.out = nn.Linear(steps * 32, num_classes)
        self.apply(_glorot_)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        for c in self.convs:
            h = F.relu(c(h))
        h = (F.max_pool1d(h, 2) + F.avg_pool1d(h, 2)) / 2.0      # MMA
        h = F.relu(self.dense(h.transpose(1, 2)))                  # (B, steps, 32)
        h = self.drop1(h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)             # BN over channel axis (Keras axis=-1)
        h = self.drop2(h).flatten(1)
        return self.out(h)

    def reg_loss(self):
        return torch.zeros((), device=self.out.weight.device)


CNN_MODELS = {"cnn_baseline": CNNBaseline, "scm_mma": SCMMMA}


# ---------------------------------------------------------------- Keras text tokenizer port
KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


class KerasLikeTokenizer:
    """tf.keras Tokenizer(num_words=N) + pad_sequences(maxlen) with Keras defaults
    (lower=True, default filters, split=' ', no OOV token, padding='pre', truncating='pre')."""

    def __init__(self, num_words=50000):
        self.num_words = num_words
        self.word_index = {}
        self._tr = str.maketrans({c: " " for c in KERAS_FILTERS})

    def _split(self, t):
        return [w for w in t.lower().translate(self._tr).split(" ") if w]

    def fit(self, texts):
        from collections import OrderedDict
        counts = OrderedDict()
        for t in texts:
            for w in self._split(t):
                counts[w] = counts.get(w, 0) + 1
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)   # stable, as Keras
        self.word_index = {w: i + 1 for i, (w, _) in enumerate(order)}
        return self

    @property
    def vocab_size(self):
        return min(len(self.word_index) + 1, self.num_words)

    def encode(self, texts, max_len):
        out = []
        for t in texts:
            seq = [self.word_index[w] for w in self._split(t)
                   if w in self.word_index and self.word_index[w] < self.num_words]
            seq = seq[-max_len:]
            out.append([0] * (max_len - len(seq)) + seq)
        return torch.tensor(out, dtype=torch.long)
