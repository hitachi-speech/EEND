# Modify this file according to a job scheduling system in your cluster.
# For more information about cmd.sh see http://kaldi-asr.org/doc/queue.html.
#
# If you use your local machine, use "run.pl".
# export train_cmd="run.pl"
# export infer_cmd="run.pl"
# export simu_cmd="run.pl"

# If you use Grid Engine, use "queue.pl"
export train_cmd="queue.pl --mem 16G -l 'hostname=c*'"
export infer_cmd="queue.pl --mem 16G -l 'hostname=c*'"
export simu_cmd="queue.pl"

# If you use SLURM, use "slurm.pl".
# export train_cmd="slurm.pl"
# export infer_cmd="slurm.pl"
# export simu_cmd="slurm.pl"
