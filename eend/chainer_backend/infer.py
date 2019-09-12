#!/usr/bin/env python3
#
# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
import os
import h5py
import numpy as np
import chainer
from chainer import Variable
from chainer import serializers
from scipy.ndimage import shift
from eend.chainer_backend.models import BLSTMDiarization
from eend.chainer_backend.models import TransformerDiarization
from eend.chainer_backend.utils import use_single_gpu
from eend import feature
from eend import kaldi_data
from eend import system_info


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


def infer(args):
    system_info.print_system_info()

    # Prepare model
    in_size = feature.get_input_dim(
            args.frame_size,
            args.context_size,
            args.input_transform)

    if args.model_type == "BLSTM":
        model = BLSTMDiarization(
                in_size=in_size,
                n_speakers=args.num_speakers,
                hidden_size=args.hidden_size,
                n_layers=args.num_lstm_layers,
                embedding_layers=args.embedding_layers,
                embedding_size=args.embedding_size)
    elif args.model_type == 'Transformer':
        model = TransformerDiarization(
                args.num_speakers,
                in_size,
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=0)
    else:
        raise ValueError('Unknown model type.')

    serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        gpuid = use_single_gpu()
        model.to_gpu()

    kaldi_obj = kaldi_data.KaldiData(args.data_dir)
    for recid in kaldi_obj.wavs:
        data, rate = kaldi_obj.load_wav(recid)
        Y = feature.stft(data, args.frame_size, args.frame_shift)
        Y = feature.transform(Y, transform_type=args.input_transform)
        Y = feature.splice(Y, context_size=args.context_size)
        Y = Y[::args.subsampling]
        out_chunks = []
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            hs = None
            for start, end in _gen_chunk_indices(len(Y), args.chunk_size):
                Y_chunked = Variable(Y[start:end])
                if args.gpu >= 0:
                    Y_chunked.to_gpu(gpuid)
                hs, ys = model.estimate_sequential(hs, [Y_chunked])
                if args.gpu >= 0:
                    ys[0].to_cpu()
                out_chunks.append(ys[0].data)
                if args.save_attention_weight == 1:
                    att_fname = f"{recid}_{start}_{end}.att.npy"
                    att_path = os.path.join(args.out_dir, att_fname)
                    model.save_attention_weight(att_path)
        outfname = recid + '.h5'
        outpath = os.path.join(args.out_dir, outfname)
        if hasattr(model, 'label_delay'):
            outdata = shift(np.vstack(out_chunks), (-model.label_delay, 0))
        else:
            outdata = np.vstack(out_chunks)
        with h5py.File(outpath, 'w') as wf:
            wf.create_dataset('T_hat', data=outdata)
