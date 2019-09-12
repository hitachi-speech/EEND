#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
stage=0

# The datasets for training must be formatted as kaldi data directory.
# Also, make sure the audio files in wav.scp are 'regular' wav files.
# Including piped commands in wav.scp makes training very slow
train_set=data/simu/data/swb_sre_tr_ns2_beta2_100000
valid_set=data/simu/data/swb_sre_cv_ns2_beta2_500
adapt_set=data/eval/callhome1_spk2
adapt_valid_set=data/eval/callhome2_spk2

# Base config files for {train,infer}.py
train_config=conf/train.yaml
infer_config=conf/infer.yaml
adapt_config=conf/adapt.yaml

# Additional arguments passed to {train,infer}.py.
# You need not edit the base config files above
train_args=
infer_args=
adapt_args=

# Model averaging options
average_start=91
average_end=100

# Adapted model averaging options
adapt_average_start=91
adapt_average_end=100

# Resume training from snapshot at this epoch
# TODO: not tested
resume=-1

# Debug purpose
debug=

. path.sh
. cmd.sh
. parse_options.sh || exit

set -eu

if [ "$debug" != "" ]; then
    # debug mode
    train_set=data/simu/data/swb_sre_tr_ns2_beta2_1000
    train_config=conf/debug/train.yaml
    average_start=3
    average_end=5
    adapt_config=conf/debug/adapt.yaml
    adapt_average_start=6
    adapt_average_end=10
fi

# Parse the config file to set bash variables like: $train_frame_shift, $infer_gpu
eval `yaml2bash.py --prefix train $train_config`
eval `yaml2bash.py --prefix infer $infer_config`

# Append gpu reservation flag to the queuing command
if [ $train_gpu -le 0 ]; then
    train_cmd+=" --gpu 1"
fi
if [ $infer_gpu -le 0 ]; then
    infer_cmd+=" --gpu 1"
fi

# Build directry names for an experiment
#  - Training
#     exp/diarize/model/{train_id}.{valid_id}.{train_config_id}
#  - Decoding
#     exp/diarize/infer/{train_id}.{valid_id}.{train_config_id}.{infer_config_id}
#  - Scoring
#     exp/diarize/scoring/{train_id}.{valid_id}.{train_config_id}.{infer_config_id}
#  - Adapation from non-adapted averaged model
#     exp/diarize/model/{train_id}.{valid_id}.{train_config_id}.{avgid}.{adapt_config_id}
train_id=$(basename $train_set)
valid_id=$(basename $valid_set)
train_config_id=$(echo $train_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
infer_config_id=$(echo $infer_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')
adapt_config_id=$(echo $adapt_config | sed -e 's%conf/%%' -e 's%/%_%' -e 's%\.yaml$%%')

# Additional arguments are added to config_id
train_config_id+=$(echo $train_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
infer_config_id+=$(echo $infer_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')
adapt_config_id+=$(echo $adapt_args | sed -e 's/\-\-/_/g' -e 's/=//g' -e 's/ \+//g')

model_id=$train_id.$valid_id.$train_config_id
model_dir=exp/diarize/model/$model_id
if [ $stage -le 1 ]; then
    echo "training model at $model_dir."
    if [ -d $model_dir ]; then
        echo "$model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $train_config \
            $train_args \
            $train_set $valid_set $model_dir \
            || exit 1
fi

ave_id=avg${average_start}-${average_end}
if [ $stage -le 2 ]; then
    echo "averaging model parameters into $model_dir/$ave_id.nnet.npz"
    if [ -s $model_dir/$ave_id.nnet.npz ]; then
        echo "$model_dir/$ave_id.nnet.npz already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $model_dir/snapshot_epoch-{$average_start..$average_end}`
    model_averaging.py $model_dir/$ave_id.nnet.npz $models || exit 1
fi

infer_dir=exp/diarize/infer/$model_id.$ave_id.$infer_config_id
if [ $stage -le 3 ]; then
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $infer_cmd $work/infer.log \
            infer.py \
            -c $infer_config \
            $infer_args \
            data/eval/$dset \
            $model_dir/$ave_id.nnet.npz \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id.$ave_id.$infer_config_id
if [ $stage -le 4 ]; then
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
        md-eval.pl -c 0.25 \
            -r data/eval/$dset/rttm \
            -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
fi

adapt_model_dir=exp/diarize/model/$model_id.$ave_id.$adapt_config_id
if [ $stage -le 5 ]; then
    echo "adapting model at $adapt_model_dir"
    if [ -d $adapt_model_dir ]; then
        echo "$adapt_model_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    work=$adapt_model_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log \
        train.py \
            -c $adapt_config \
            $adapt_args \
            --initmodel $model_dir/$ave_id.nnet.npz \
            $adapt_set $adapt_valid_set $adapt_model_dir \
                || exit 1
fi

adapt_ave_id=avg${adapt_average_start}-${adapt_average_end}
if [ $stage -le 6 ]; then
    echo "averaging models into $adapt_model_dir/$adapt_ave_id.nnet.gz"
    if [ -s $adapt_model_dir/$adapt_ave_id.nnet.npz ]; then
        echo "$adapt_model_dir/$adapt_ave_id.nnet.npz already exists."
        echo " if you want to retry, please remove it."
        exit 1
    fi
    models=`eval echo $adapt_model_dir/snapshot_epoch-{$adapt_average_start..$adapt_average_end}`
    model_averaging.py $adapt_model_dir/$adapt_ave_id.nnet.npz $models || exit 1
fi

infer_dir=exp/diarize/infer/$model_id.$ave_id.$adapt_config_id.$adapt_ave_id.$infer_config_id
if [ $stage -le 7 ]; then
    echo "inference at $infer_dir"
    if [ -d $infer_dir ]; then
        echo "$infer_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$infer_dir/$dset/.work
        mkdir -p $work
        $train_cmd $work/infer.log \
            infer.py -c $infer_config \
            data/eval/${dset} \
            $adapt_model_dir/$adapt_ave_id.nnet.npz \
            $infer_dir/$dset \
            || exit 1
    done
fi

scoring_dir=exp/diarize/scoring/$model_id.$ave_id.$adapt_config_id.$adapt_ave_id.$infer_config_id
if [ $stage -le 8 ]; then
    echo "scoring at $scoring_dir"
    if [ -d $scoring_dir ]; then
        echo "$scoring_dir already exists. "
        echo " if you want to retry, please remove it."
        exit 1
    fi
    for dset in callhome2_spk2; do
        work=$scoring_dir/$dset/.work
        mkdir -p $work
        find $infer_dir/$dset -iname "*.h5" > $work/file_list_$dset
        for med in 1 11; do
        for th in 0.3 0.4 0.5 0.6 0.7; do
        make_rttm.py --median=$med --threshold=$th \
            --frame_shift=$infer_frame_shift --subsampling=$infer_subsampling --sampling_rate=$infer_sampling_rate \
            $work/file_list_$dset $scoring_dir/$dset/hyp_${th}_$med.rttm
        md-eval.pl -c 0.25 \
            -r data/eval/$dset/rttm \
            -s $scoring_dir/$dset/hyp_${th}_$med.rttm > $scoring_dir/$dset/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
        done
        done
    done
fi

if [ $stage -le 9 ]; then
    for dset in callhome2_spk2; do
        best_score.sh $scoring_dir/$dset
    done
fi
echo "Finished !"

