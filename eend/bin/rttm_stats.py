#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import numpy as np
import argparse

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
        rttm, start=0, end=None, rate=16000, shift=256):
    """ Get frame labels from RTTM file
    Args:
        start: start time in seconds
        end: end time in seconds
        rate: sampling rate
        shift: number of frame shift samples
        n_speakers: number of speakers
            if None, determined from rttm file
    Returns:
        labels.T: frame labels
            (n_frames, n_speaker)-shaped numpy.int32 array
        speakers: list of speaker ids
    """
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
        if sf is not None or ef is not None:
            labels[sf:ef, seg_sp] = 1
    return labels.T, speakers


parser = argparse.ArgumentParser()
parser.add_argument('rttm')
args = parser.parse_args()
rttm = load_rttm(args.rttm)

def _min_max_ave(a):
    return [f(a) for f in [np.min, np.max, np.mean]]

vafs = []
uds = []
ids = []
reclens = []
pres = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
den = 0
recordings = np.unique(rttm['recid'])
for recid in recordings:
    rec = rttm[rttm['recid'] == recid]
    speakers = np.unique(rec['speaker'])
    for speaker in speakers:
        spk = rec[rec['speaker'] == speaker]
        spk.sort()
        durs = spk['et'] - spk['st']
        stats_dur = _min_max_ave(durs)
        uds.append(np.mean(durs))
        if len(durs) > 1:
            intervals = spk['st'][1:] - spk['et'][:-1]
            stats_int = _min_max_ave(intervals)
            ids.append(np.mean(intervals))
            vafs.append(np.sum(durs)/(np.sum(durs) + np.sum(intervals)))
    labels, _ = get_frame_labels(rec)
    n_presense = np.sum(labels, axis=0)
    for n in np.unique(n_presense):
        pres[n] += np.sum(n_presense == n)
    den += len(n_presense)
    #for s in speakers: print(s)
    reclens.append(rec['et'].max() - rec['st'].min())

print(list(range(2, len(pres))))
total_speaker = np.sum([n * pres[n] for n in range(len(pres))])
total_overlap = np.sum([n * pres[n] for n in range(2, len(pres))])
print(total_speaker, total_overlap, total_overlap/total_speaker)
print("single-speaker overlap", pres[3]/np.sum(pres[2:]))
print(len(recordings), np.mean(reclens), np.mean(vafs), np.mean(uds), np.mean(ids), "overlap ratio:", np.sum(pres[2:])/np.sum(pres[1:]), "overlaps", ' '.join(str(x) for x in pres/den))
