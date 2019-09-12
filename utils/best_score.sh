grep OVER $1/result_th0.[^_]*_med[^_]*_collar0.25 | grep -v nooverlap | sort -nrk 7 | tail -n 1
