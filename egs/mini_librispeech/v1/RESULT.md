
# Training curve
 GPU: Tesla K80

```
 grep loss exp/diarize/model/train_clean_5_ns2_beta2_500.dev_clean_2_ns2_beta2_500.train/log
        "main/loss": 0.8094629645347595,
        "validation/main/loss": 0.7502496838569641,
        "main/loss": 0.6841621398925781,
        "validation/main/loss": 0.6442975997924805,
        "main/loss": 0.5633799433708191,
        "validation/main/loss": 0.5978456139564514,
        "main/loss": 0.5073038339614868,
        "validation/main/loss": 0.5673154592514038,
        "main/loss": 0.47916650772094727,
        "validation/main/loss": 0.5508813261985779,
        "main/loss": 0.46045243740081787,
        "validation/main/loss": 0.53536057472229,
        "main/loss": 0.44897904992103577,
        "validation/main/loss": 0.5264081358909607,
        "main/loss": 0.4393312335014343,
        "validation/main/loss": 0.520709216594696,
        "main/loss": 0.4310261905193329,
        "validation/main/loss": 0.510313093662262,
        "main/loss": 0.42343708872795105,
        "validation/main/loss": 0.5055857300758362,
```

# Final DER

```
exp/diarize/scoring/train_clean_5_ns2_beta2_500.dev_clean_2_ns2_beta2_500.train.avg8-10.infer/dev_clean_2_ns2_beta2_500/result_th0.7_med11_collar0.25: OVERALL SPEAKER DIARIZATION ERROR = 29.96 percent of scored speaker time  `(ALL)
```
