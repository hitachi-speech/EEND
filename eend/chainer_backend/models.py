# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from itertools import permutations
from chainer import cuda
from chainer import reporter
from chainer import configuration
from eend.chainer_backend.transformer import TransformerEncoder
from eend.chainer_backend.encoder_decoder_attractor import EncoderDecoderAttractor

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


def batch_pit_loss_faster(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions
      ts: B-length list of labels
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """

    n_speakers = ts[0].shape[1]
    xp = chainer.backend.get_array_module(ys[0])
    # (B, T, C)
    ys = F.pad_sequence(ys, padding=-1)

    losses = []
    for shift in range(n_speakers):
        # rolled along with speaker-axis
        ts_roll = [xp.roll(t, -shift, axis=1) for t in ts]
        ts_roll = F.pad_sequence(ts_roll, padding=-1)
        # loss: (B, T, C)
        loss = F.sigmoid_cross_entropy(ys, ts_roll, reduce='no')
        # sum over time: (B, C)
        loss = F.sum(loss, axis=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = F.stack(losses, axis=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = xp.array(
        list(permutations(range(n_speakers))),
        dtype='i',
    )
    # y_inds: [0,1,2,3]
    y_ind = xp.arange(n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = xp.mod(perms - y_ind, n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            F.mean(losses[:, y_ind, t_ind], axis=1))
    # losses_perm: (B, Perm)
    losses_perm = F.stack(losses_perm, axis=1)

    min_loss = F.sum(F.min(losses_perm, axis=1))

    min_loss = F.sum(F.min(losses_perm, axis=1))
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = xp.argmin(losses_perm.array, axis=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]

    return min_loss, labels_perm


def standard_loss(ys, ts, label_delay=0):
    losses = [F.sigmoid_cross_entropy(y, t) * len(y) for y, t in zip(ys, ts)]
    loss = F.sum(F.stack(losses))
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss


def batch_pit_n_speaker_loss(ys, ts, n_speakers_list):
    """
    PIT loss over mini-batch.
    Args:
      ys: B-length list of predictions (pre-activations)
      ts: B-length list of labels
      n_speakers_list: list of n_speakers in batch
    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    max_n_speakers = ts[0].shape[1]
    xp = chainer.backend.get_array_module(ys[0])
    # (B, T, C)
    ys = F.pad_sequence(ys, padding=-1)

    losses = []
    for shift in range(max_n_speakers):
        # rolled along with speaker-axis
        ts_roll = [xp.roll(t, -shift, axis=1) for t in ts]
        ts_roll = F.pad_sequence(ts_roll, padding=-1)
        # loss: (B, T, C)
        loss = F.sigmoid_cross_entropy(ys, ts_roll, reduce='no')
        # sum over time: (B, C)
        loss = F.sum(loss, axis=1)
        losses.append(loss)
    # losses: (B, C, C)
    losses = F.stack(losses, axis=2)
    # losses[b, i, j] is a loss between
    # `i`-th speaker in y and `(i+j)%C`-th speaker in t

    perms = xp.array(
        list(permutations(range(max_n_speakers))),
        dtype='i',
    )
    # y_ind: [0,1,2,3]
    y_ind = xp.arange(max_n_speakers, dtype='i')
    #  perms  -> relation to t_inds      -> t_inds
    # 0,1,2,3 -> 0+j=0,1+j=1,2+j=2,3+j=3 -> 0,0,0,0
    # 0,1,3,2 -> 0+j=0,1+j=1,2+j=3,3+j=2 -> 0,0,1,3
    t_inds = xp.mod(perms - y_ind, max_n_speakers)

    losses_perm = []
    for t_ind in t_inds:
        losses_perm.append(
            F.mean(losses[:, y_ind, t_ind], axis=1))
    # losses_perm: (B, Perm)
    losses_perm = F.stack(losses_perm, axis=1)

    # masks: (B, Perms)
    def select_perm_indices(num, max_num):
        perms = list(permutations(range(max_num)))
        sub_perms = list(permutations(range(num)))
        return [
            [x[:num] for x in perms].index(perm)
            for perm in sub_perms]
    masks = xp.full_like(losses_perm.array, xp.inf)
    for i, t in enumerate(ts):
        n_speakers = n_speakers_list[i]
        indices = select_perm_indices(n_speakers, max_n_speakers)
        masks[i, indices] = 0
    losses_perm += masks

    min_loss = F.sum(F.min(losses_perm, axis=1))
    n_frames = np.sum([t.shape[0] for t in ts])
    min_loss = min_loss / n_frames

    min_indices = xp.argmin(losses_perm.array, axis=1)
    labels_perm = [t[:, perms[idx]] for t, idx in zip(ts, min_indices)]
    labels_perm = [t[:, :n_speakers] for t, n_speakers in zip(labels_perm, n_speakers_list)]

    return min_loss, labels_perm


def add_silence_labels(ts):
    xp = cuda.get_array_module(ts[0])
    # pad label's speaker-dim to be model's n_speakers
    for i, t in enumerate(ts):
        ts[i] = xp.pad(
            t,
            [(0, 0), (0, 1)],
            mode='constant',
            constant_values=0.,
        )
    return ts


def pad_labels(ts, out_size):
    xp = cuda.get_array_module(ts[0])
    # pad label's speaker-dim to be model's n_speakers
    for i, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts[i] = xp.pad(
                t,
                [(0, 0), (0, out_size - t.shape[1])],
                mode='constant',
                constant_values=0.,
            )
        elif t.shape[1] > out_size:
            # truncate
            raise ValueError
    return ts


def pad_results(ys, out_size):
    xp = cuda.get_array_module(ys[0])
    # pad label's speaker-dim to be model's n_speakers
    ys_padded = []
    for i, y in enumerate(ys):
        if y.shape[1] < out_size:
            # padding
            ys_padded.append(F.concat([y, chainer.Variable(xp.zeros((y.shape[0], out_size - y.shape[1]), dtype=y.dtype))], axis=1))
        elif y.shape[1] > out_size:
            # truncate
            raise ValueError
        else:
            ys_padded.append(y)
    return ys_padded


def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (ndarray): (T,C)-shaped pre-activation values
      label (ndarray): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    xp = chainer.backend.get_array_module(pred)
    label = label[:len(label) - label_delay, ...]
    decisions = F.sigmoid(pred[label_delay:, ...]).array > 0.5
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
      ys: B-length list of predictions (Variable)
      labels: B-length list of labels (ndarray)
      observer: target link (chainer.Chain)
    """
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y.array, t)
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

    def estimate_sequential(self, hx, xs, **kwargs):
        ys = self.forward(xs, activation=F.sigmoid)
        return None, ys

    def __call__(self, xs, ts):
        ys = self.forward(xs)
        # loss, labels = batch_pit_loss_faster(ys, ts)
        n_speakers = [t.shape[1] for t in ts]
        loss, labels = batch_pit_n_speaker_loss(ys, ts, n_speakers)
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


class TransformerEDADiarization(chainer.Chain):

    def __init__(self, in_size, n_units, n_heads, n_layers, dropout,
                 attractor_loss_ratio=1.0,
                 attractor_encoder_dropout=0.1,
                 attractor_decoder_dropout=0.1):
        """ Self-attention-based diarization model.

        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        super(TransformerEDADiarization, self).__init__()
        with self.init_scope():
            self.enc = TransformerEncoder(
                in_size, n_layers, n_units, h=n_heads
            )
            self.eda = EncoderDecoderAttractor(
                n_units,
                encoder_dropout=attractor_encoder_dropout,
                decoder_dropout=attractor_decoder_dropout,
            )
        self.attractor_loss_ratio = attractor_loss_ratio

    def forward(self, xs, n_speakers=None, activation=None):
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        xs = F.pad_sequence(xs, padding=-1)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        ys = emb
        # emb: [(T, E), ...]
        emb = F.separate(emb.reshape(pad_shape[0], pad_shape[1], -1), axis=0)
        emb = [F.get_item(e, slice(0, ilen)) for e, ilen in zip(emb, ilens)]

        return emb

    def estimate_sequential(self, hx, xs, **kwargs):
        emb = self.forward(xs)
        ys_active = []
        n_speakers = kwargs.get('n_speakers')
        th = kwargs.get('th')
        shuffle = kwargs.get('shuffle')
        if shuffle:
            xp = cuda.get_array_module(emb[0])
            orders = [xp.arange(e.shape[0]) for e in emb]
            for order in orders:
                xp.random.shuffle(order)
            attractors, probs = self.eda.estimate([e[order] for e, order in zip(emb, orders)])
        else:
            attractors, probs = self.eda.estimate(emb)
        ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]
        ys = [F.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
            if n_speakers is not None:
                ys_active.append(y[:, :n_speakers])
            elif th is not None:
                silence = np.where(cuda.to_cpu(p.data) < th)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError('n_speakers or th has to be given.')
        return None, ys_active

    def __call__(self, xs, ts):
        n_speakers = [t.shape[1] for t in ts]
        emb = self.forward(xs, n_speakers)
        attractor_loss, attractors = self.eda(emb, n_speakers)
        # ys: [(T, C), ...]
        ys = [F.matmul(e, att, transb=True) for e, att in zip(emb, attractors)]

        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(ts, max_n_speakers)
        ys_padded = pad_results(ys, max_n_speakers)

        if configuration.config.train:
            # with chainer.using_config('enable_backprop', False):
            loss, labels = batch_pit_n_speaker_loss(ys_padded, ts_padded, n_speakers)
            loss = standard_loss(ys, labels)
        else:
            loss, labels = batch_pit_n_speaker_loss(ys_padded, ts_padded, n_speakers)
            loss = standard_loss(ys, labels)

        reporter.report({'loss': loss}, self)
        reporter.report({'attractor_loss': attractor_loss}, self)
        report_diarization_error(ys, labels, self)
        return loss + attractor_loss * self.attractor_loss_ratio

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

    def estimate_sequential(self, hx, xs, **kwargs):
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
