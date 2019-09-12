#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_rttm(rttm_file):
    """ load rttm file as numpy structured array """
    segments = []
    for line in open(rttm_file):
        toks = line.strip().split()
        # number of columns is 9 (RT-05S) or 10 (RT-09S)
        (stype, fileid, ch, start, duration,
         _, _, speaker, _) = toks[:9]
        if stype != "SPEAKER":
            continue
        start = float(start)
        end = start + float(duration)
        segments.append((fileid, speaker, start, end))
    return np.array(segments, dtype=[
        ('recid', 'object'), ('speaker', 'object'), ('st', 'f'), ('et', 'f')])


def time2frame(t, rate, shift):
    """ time in second (float) to frame index (int) """
    return np.rint(t * rate / shift).astype(int)


def get_frame_labels(
        rttm_file, recid, start=0, end=None, rate=16000, shift=256):
    """ Get frame labels from RTTM file
    Args:
        rttm_file: RTTM file
        recid: Recording id is the 2nd column of RTTM file,
               must be identical to the rec id in wav.scp
        start: start time in seconds
        end: end time in seconds
        rate: sampling rate
        shift: number of frame shift samples
    Returns:
        labels.T: frame labels
            (n_frames, n_speaker)-shaped numpy.int32 array
        speakers: list of speaker ids
    """
    rttm = load_rttm(rttm_file)
    # filter by recording id
    rttm = rttm[rttm['recid'] == recid]
    # sorted uniq speaker ids
    speakers = np.unique(rttm['speaker']).tolist()
    # start and end frames
    rec_sf = time2frame(start, rate, shift)
    rec_ef = time2frame(end if end else rttm['et'].max(), rate, shift)
    labels = np.zeros((rec_ef - rec_sf, len(speakers)), dtype=np.int32)
    for seg in rttm:
        seg_sp = speakers.index(seg['speaker'])
        seg_sf = time2frame(seg['st'], rate, shift)
        seg_ef = time2frame(seg['et'], rate, shift)
        # relative frame index from 'rec_sf'
        sf = ef = None
        if rec_sf <= seg_sf and seg_sf < rec_ef:
            sf = seg_sf - rec_sf
        if rec_sf < seg_ef and seg_ef <= rec_ef:
            ef = seg_ef - rec_sf
        if seg_sf < rec_sf and rec_ef < seg_ef:
            sf = 0
        if sf is not None or ef is not None:
            labels[sf:ef, seg_sp] = 1
    return labels.T, speakers


def attention_plot(args):
    """ Plots attention weights with reference labels.
    """
    # attention weights at specified layer
    att_w = np.load(args.att_file)[args.layer, ...]
    # extract recid from att_file name, "<recid>_<start>_<end>.att.npy"
    recid = '_'.join(os.path.basename(args.att_file).split('_')[:-2])

    start_frame = int(os.path.basename(args.att_file).split('_')[-2])
    end_frame = start_frame + att_w.shape[1]

    ref, ref_spks = get_frame_labels(
        args.rttm_file, recid, rate=args.rate,
        start=start_frame * args.shift / args.rate,
        end=end_frame * args.shift / args.rate,
        shift=args.shift)

    if args.span:
        start, end = [int(x) for x in args.span.split(":")]
    else:
        start = 0
        end = att_w.shape[2]
    n_subplots = att_w.shape[0]

    ref_height = 1

    fig, axes = plt.subplots(1, n_subplots,
                             figsize=(att_w.shape[0] * 4, 4 + ref_height))

    for i, (ax, aw) in enumerate(zip(axes, att_w)):
        divider = make_axes_locatable(ax)
        if args.ref_type == 'line':
            colors = ['k', 'r']
            # stack figures from bottom to top
            for spk, r in reversed(list(enumerate(ref))):
                ax_label = divider.append_axes('top', 0.3, pad=0.1, sharex=ax)
                ax_label.xaxis.set_tick_params(labelbottom=False)
                ax_label.xaxis.set_tick_params(bottom=False)
                ax_label.yaxis.set_tick_params(left=False)
                ax_label.set_ylim([-0.5, 1.5])
                ax_label.set_yticks(np.arange(2))
                ax_label.set_yticklabels(['silence', 'speech'])
                ax_label.set_ylabel('Spk {}'.format(spk+1),
                                    rotation=0, va='center', labelpad=15)
                if i > 0:
                    ax_label.yaxis.set_tick_params(labelleft=False)
                    ax_label.set_ylabel('')
                ax_label.plot(np.arange(r.size), r, lw=1, c=colors[spk == i])
        elif args.ref_type == 'fill':
            for spk, r in reversed(list(enumerate(ref, 1))):
                ax_label = divider.append_axes('top', 0.2, pad=0.1, sharex=ax)
                ax_label.xaxis.set_tick_params(labelbottom=False)
                ax_label.yaxis.set_tick_params(labelleft=False)
                ax_label.xaxis.set_tick_params(bottom=False, left=False)
                ax_label.yaxis.set_tick_params(left=False)
                if args.span:
                    ax_label.set_xlim([start, end])
                ax_label.set_ylabel('Spk {}'.format(spk),
                                    rotation=0, va='center', labelpad=15)
                aspect = (end - start) / 40
                ax_label.imshow(r[np.newaxis, :], aspect=aspect, cmap='binary')

        ax.imshow(aw, aspect='equal', cmap=args.colormap)
        if args.span:
            ax.set_xlim([start, end])
            ax.set_ylim([end, start])
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_xticks(ax.get_yticks()[1:-1])
        if args.invert_yaxis:
            ax.invert_yaxis()
        if args.add_title:
            ax.set_title('Head {}'.format(i + 1), y=-0.25, fontsize=16)

    if args.add_title:
        # manual spacing at bottom
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        plt.tight_layout()
    plt.savefig(args.pdf_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot attention weight')
    parser.add_argument('att_file',
                        help='Attention weight file.')
    parser.add_argument('rttm_file',
                        help='RTTM file,')
    parser.add_argument('pdf_file',
                        help='Output pdf file.')
    parser.add_argument('--colormap', default='binary',
                        help='colormap for heatmaps: gray, jet, viridis, etc.')
    parser.add_argument('--invert_yaxis', action='store_true',
                        help='invert y-axis in heatmap')
    parser.add_argument('--add_title', action='store_true',
                        help='put captions "Head N" under heatmaps')
    parser.add_argument('--ref_type', choices=['line', 'fill'], default='fill',
                        help='reference label appearance')
    parser.add_argument('--span', default='',
                        help='colon-delimited start/end frame id')
    parser.add_argument('--layer', default=1,
                        help='0-origin layer index')
    parser.add_argument('--rate', default=8000,
                        help='sampleing rate')
    parser.add_argument('--shift', default=800,
                        help='frame-shift * subsampling')
    args = parser.parse_args()

    matplotlib.pyplot.switch_backend('Agg')
    attention_plot(args)
