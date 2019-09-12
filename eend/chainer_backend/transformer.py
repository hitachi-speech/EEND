# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import numpy as np
from chainer.training import extension
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class NoamScheduler(extension.Extension):
    """ learning rate scheduler used in the transformer
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Scaling factor is implemented as in
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    """

    def __init__(self, d_model, warmup_steps, scale=1.0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.last_value = None
        self.t = 0

    def initialize(self, trainer):
        optimizer = trainer.updater.get_optimizer('main')
        if self.last_value:
            # resume
            setattr(optimizer, 'alpha', self.last_value)
        else:
            # the initiallearning rate is set as step = 1,
            init_value = self.scale * self.d_model ** (-0.5) * \
                self.warmup_steps ** (-1.5)
            setattr(optimizer, 'alpha', init_value)
            self.last_value = init_value

    def __call__(self, trainer):
        self.t += 1
        optimizer = trainer.updater.get_optimizer('main')
        value = self.scale * self.d_model ** (-0.5) * \
            min(self.t ** (-0.5), self.t * self.warmup_steps ** (-1.5))
        setattr(optimizer, 'alpha', value)
        self.last_value = value

    def serialize(self, serializer):
        self.t = serializer('t', self.t)
        self.last_value = serializer('last_value', self.last_value)


class MultiHeadSelfAttention(Chain):

    """ Multi head "self" attention layer
    """

    def __init__(self, n_units, h=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        with self.init_scope():
            self.linearQ = L.Linear(n_units, n_units)
            self.linearK = L.Linear(n_units, n_units)
            self.linearV = L.Linear(n_units, n_units)
            self.linearO = L.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        # attention for plot
        self.att = None

    def __call__(self, x, batch_size):
        # x: (BT, F)
        # TODO: if chainer >= 5.0, use linear functions with 'n_batch_axes'
        # and x be (B, T, F), then remove batch_size.
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = F.matmul(
            F.swapaxes(q, 1, 2), k.transpose(0, 2, 3, 1)) / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, axis=3)
        p_att = F.dropout(self.att, self.dropout)
        x = F.matmul(p_att, F.swapaxes(v, 1, 2))
        x = F.swapaxes(x, 1, 2).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(Chain):

    """ Positionwise feed-forward layer
    """

    def __init__(self, n_units, d_units, dropout):
        super(PositionwiseFeedForward, self).__init__()
        with self.init_scope():
            self.linear1 = L.Linear(n_units, d_units)
            self.linear2 = L.Linear(d_units, n_units)
            self.dropout = dropout

    def __call__(self, x):
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))


class PositionalEncoding(Chain):

    """ Positional encoding function
    """

    def __init__(self, n_units, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        positions = np.arange(0, max_len, dtype='f')[:, None]
        dens = np.exp(
            np.arange(0, n_units, 2, dtype='f') * -(np.log(10000.) / n_units))
        self.enc = np.zeros((max_len, n_units), dtype='f')
        self.enc[:, ::2] = np.sin(positions * dens)
        self.enc[:, 1::2] = np.cos(positions * dens)
        self.scale = np.sqrt(n_units)

    def __call__(self, x):
        x = x * self.scale + self.xp.array(self.enc[:, :x.shape[1]])
        return F.dropout(x, self.dropout)


class TransformerEncoder(Chain):
    def __init__(self, idim, n_layers, n_units,
                 e_units=2048, h=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        with self.init_scope():
            self.linear_in = L.Linear(idim, n_units)
            self.lnorm_in = L.LayerNormalization(n_units)
            self.pos_enc = PositionalEncoding(n_units, dropout, 5000)
            self.n_layers = n_layers
            self.dropout = dropout
            for i in range(n_layers):
                setattr(self, '{}{:d}'.format("lnorm1_", i),
                        L.LayerNormalization(n_units))
                setattr(self, '{}{:d}'.format("self_att_", i),
                        MultiHeadSelfAttention(n_units, h))
                setattr(self, '{}{:d}'.format("lnorm2_", i),
                        L.LayerNormalization(n_units))
                setattr(self, '{}{:d}'.format("ff_", i),
                        PositionwiseFeedForward(n_units, e_units, dropout))
            self.lnorm_out = L.LayerNormalization(n_units)

    def __call__(self, x):
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)
