#!/usr/bin/env python3

# Copyright 2020 Hitachi, Ltd. (author: Shota Horiguchi)
# Licensed under the MIT license.

from chainer import Chain, cuda
import chainer.functions as F
import chainer.links as L


class EncoderDecoderAttractor(Chain):

    def __init__(self, n_units, encoder_dropout=0.1, decoder_dropout=0.1):
        super(EncoderDecoderAttractor, self).__init__()
        with self.init_scope():
            self.encoder = L.NStepLSTM(1, n_units, n_units, encoder_dropout)
            self.decoder = L.NStepLSTM(1, n_units, n_units, decoder_dropout)
            self.counter = L.Linear(n_units, 1)
        self.n_units = n_units

    def forward(self, xs, zeros):
        hx, cx, _ = self.encoder(None, None, xs)
        _, _, attractors = self.decoder(hx, cx, zeros)
        return attractors

    def estimate(self, xs, max_n_speakers=15):
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers

        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
        Returns:
          attractors: List of (N,D)-shaped attractors
          probs: List of attractor existence probabilities
        """

        xp = cuda.get_array_module(xs[0])
        zeros = [xp.zeros((max_n_speakers, self.n_units), dtype=xp.float32) for _ in xs]
        attractors = self.forward(xs, zeros)
        probs = [F.sigmoid(F.flatten(self.counter(att))) for att in attractors]
        return attractors, probs

    def __call__(self, xs, n_speakers):
        """
        Calculate attractors from embedding sequences with given number of speakers

        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number of speakers is unknown (ex. test phase)
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """

        xp = cuda.get_array_module(xs[0])
        zeros = [xp.zeros((n_spk + 1, self.n_units), dtype=xp.float32) for n_spk in n_speakers]
        attractors = self.forward(xs, zeros)
        labels = F.concat([xp.array([[1] * n_spk + [0]], xp.int32) for n_spk in n_speakers], axis=1)
        logit = F.concat([F.reshape(self.counter(att), (-1, n_spk + 1)) for att, n_spk in zip(attractors, n_speakers)], axis=1)
        loss = F.sigmoid_cross_entropy(logit, labels)

        # The final attractor does not correspond to a speaker so remove it
        # attractors = [att[:-1] for att in attractors]
        attractors = [att[slice(0, att.shape[0] - 1)] for att in attractors]
        return loss, attractors
