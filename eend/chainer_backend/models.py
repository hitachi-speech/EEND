# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from itertools import permutations
from chainer import cuda
from chainer import reporter
from eend.chainer_backend.transformer import TransformerEncoder

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                   in permutations(range(label.shape[-1]))]
    losses = F.stack(
        [F.sigmoid_cross_entropy(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    xp = cuda.get_array_module(losses)
    min_loss = F.min(losses) * (len(label) - label_delay)
    min_index = cuda.to_cpu(xp.argmin(losses.data))

    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = F.sum(F.stack(losses))
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels


def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred: (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    xp = cuda.get_array_module(pred)
    label = label[:len(label) - label_delay, ...].data
    decisions = F.sigmoid(pred[label_delay:, ...]).data > 0.5
    n_ref = xp.sum(label, axis=-1)
    n_sys = xp.sum(decisions, axis=-1)
    res = {}
    res['speech_scored'] = xp.sum(n_ref > 0)
    res['speech_miss'] = xp.sum(
        xp.logical_and(n_ref > 0, n_sys == 0))
    res['speech_falarm'] = xp.sum(
        xp.logical_and(n_ref == 0, n_sys > 0))
    res['speaker_scored'] = xp.sum(n_ref)
    res['speaker_miss'] = xp.sum(xp.maximum(n_ref - n_sys, 0))
    res['speaker_falarm'] = xp.sum(xp.maximum(n_sys - n_ref, 0))
    n_map = xp.sum(
        xp.logical_and(label == 1, decisions == 1),
        axis=-1)
    res['speaker_error'] = xp.sum(xp.minimum(n_ref, n_sys) - n_map)
    res['correct'] = xp.sum(label == decisions) / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res


def report_diarization_error(ys, labels, observer):
    """
    Reports diarization errors using chainer.reporter

    Args:
      ys: B-length list of predictions
      labels: B-length list of labels
      observer: target link (chainer.Chain)
    """
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y, t)
        for key in stats:
            reporter.report({key: stats[key]}, observer)


def dc_loss(embedding, label):
    """
    Deep clustering loss function.

    Args:
      embedding: (T,D)-shaped activation values
      label: (T,C)-shaped labels
    return:
      (1,)-shaped squared flobenius norm of the difference
      between embedding and label affinity matrices
    """
    xp = cuda.get_array_module(label)
    b = xp.zeros((label.shape[0], 2**label.shape[1]))
    b[np.arange(label.shape[0]),
      [int(''.join(str(x) for x in t), base=2) for t in label.data]] = 1

    label_f = chainer.Variable(b.astype(np.float32))
    loss = F.sum(F.square(F.matmul(embedding, embedding, True, False))) \
        + F.sum(F.square(F.matmul(label_f, label_f, True, False))) \
        - 2 * F.sum(F.square(F.matmul(embedding, label_f, True, False)))
    return loss


class TransformerDiarization(chainer.Chain):
    def __init__(self,
                 n_speakers,
                 in_size,
                 n_units,
                 n_heads,
                 n_layers,
                 dropout
                 ):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerDiarization, self).__init__()
        with self.init_scope():
            self.enc = TransformerEncoder(
                in_size, n_layers, n_units, h=n_heads)
            self.linear = L.Linear(n_units, n_speakers)

    def forward(self, xs, activation=None):
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        xs = F.pad_sequence(xs, padding=-1)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # ys: (B*T, C)
        ys = self.linear(emb)
        if activation:
            ys = activation(ys)
        # ys: [(T, C), ...]
        ys = F.separate(ys.reshape(pad_shape[0], pad_shape[1], -1), axis=0)
        ys = [F.get_item(y, slice(0, ilen)) for y, ilen in zip(ys, ilens)]
        return ys

    def estimate_sequential(self, hx, xs):
        ys = self.forward(xs, activation=F.sigmoid)
        return None, ys

    def __call__(self, xs, ts):
        ys = self.forward(xs)
        loss, labels = batch_pit_loss(ys, ts)
        reporter.report({'loss': loss}, self)
        report_diarization_error(ys, labels, self)
        return loss

    def save_attention_weight(self, ofile, batch_index=0):
        att_weights = []
        for l in range(self.enc.n_layers):
            att_layer = getattr(self.enc, f'self_att_{l}')
            # att.shape is (B, h, T, T); pick the first sample in batch
            att_w = att_layer.att[batch_index, ...]
            att_w.to_cpu()
            att_weights.append(att_w.data)
        # save as (n_layers, h, T, T)-shaped arryay
        np.save(ofile, np.array(att_weights))


class BLSTMDiarization(chainer.Chain):
    def __init__(self,
                 n_speakers=4,
                 dropout=0.25,
                 in_size=513,
                 hidden_size=256,
                 n_layers=1,
                 embedding_layers=1,
                 embedding_size=20,
                 dc_loss_ratio=0.5,
                 ):
        """ BLSTM-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          dropout (float): dropout ratio
          in_size (int): Dimension of input feature vector
          hidden_size (int): Number of hidden units in LSTM
          n_layers (int): Number of LSTM layers after embedding
          embedding_layers (int): Number of LSTM layers for embedding
          embedding_size (int): Dimension of embedding vector
          dc_loss_ratio (float): mixing parameter for DPCL loss
        """
        super(BLSTMDiarization, self).__init__()
        with self.init_scope():
            self.bi_lstm1 = L.NStepBiLSTM(
                n_layers, hidden_size * 2, hidden_size, dropout)
            self.bi_lstm_emb = L.NStepBiLSTM(
                embedding_layers, in_size, hidden_size, dropout)
            self.linear1 = L.Linear(hidden_size * 2, n_speakers)
            self.linear2 = L.Linear(hidden_size * 2, embedding_size)
        self.dc_loss_ratio = dc_loss_ratio
        self.n_speakers = n_speakers

    def forward(self, xs, hs=None, activation=None):
        if hs is not None:
            hx1, cx1, hx_emb, cx_emb = hs
        else:
            hx1 = cx1 = hx_emb = cx_emb = None
        # forward to LSTM layers
        hy_emb, cy_emb, ems = self.bi_lstm_emb(hx_emb, cx_emb, xs)
        hy1, cy1, ys = self.bi_lstm1(hx1, cx1, ems)
        # main branch
        ys_stack = F.vstack(ys)
        ys = self.linear1(ys_stack)
        if activation:
            ys = activation(ys)
        ilens = [x.shape[0] for x in xs]
        ys = F.split_axis(ys, np.cumsum(ilens[:-1]), axis=0)
        # embedding branch
        ems_stack = F.vstack(ems)
        ems = F.normalize(F.tanh(self.linear2(ems_stack)))
        ems = F.split_axis(ems, np.cumsum(ilens[:-1]), axis=0)

        if not isinstance(ys, tuple):
            ys = [ys]
            ems = [ems]
        return [hy1, cy1, hy_emb, cy_emb], ys, ems

    def estimate_sequential(self, hx, xs):
        hy, ys, ems = self.forward(xs, hx, activation=F.sigmoid)
        return hy, ys

    def __call__(self, xs, ts):
        _, ys, ems = self.forward(xs)
        # PIT loss
        loss, labels = batch_pit_loss(ys, ts)
        reporter.report({'loss_pit': loss}, self)
        report_diarization_error(ys, labels, self)
        # DPCL loss
        loss_dc = F.sum(
            F.stack([dc_loss(em, t) for (em, t) in zip(ems, ts)]))
        n_frames = np.sum([t.shape[0] for t in ts])
        loss_dc = loss_dc / (n_frames ** 2)
        reporter.report({'loss_dc': loss_dc}, self)
        # Multi-objective
        loss = (1 - self.dc_loss_ratio) * loss + self.dc_loss_ratio * loss_dc
        reporter.report({'loss': loss}, self)

        return loss
