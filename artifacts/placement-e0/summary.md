# Experiment C E0 results

- Cases: 31
- Arm A reached arithmetic floor: 29/31
- Arm B reached target: 31/31
- Median Arm A time-to-floor upper bound: 1.0 s

| Case | m | Floor | Old greedy | Old MILP | Arm A time-to-floor <= | Arm B spread | Arm B time | Proof |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| depth-p30-s00-l12 | 4 | 128 | 1024.0 | 128.0 | 1.0 | 128.0 | 0.580166 | arithmetic_lower_bound |
| depth-p30-s00-l16 | 4 | 128 | 512.0 | 128.0 | 2.0 | 128.0 | 0.686055 | arithmetic_lower_bound |
| depth-p30-s00-l24 | 4 | 128 | 768.0 | 128.0 | 1.0 | 128.0 | 0.098574 | arithmetic_lower_bound |
| depth-p30-s00-l32 | 4 | 128 | 256.0 | 128.0 | 0.5 | 128.0 | 0.108745 | arithmetic_lower_bound |
| depth-p30-s00-l48 | 4 | 128 | 256.0 | 128.0 | 1.0 | 128.0 | 0.368254 | arithmetic_lower_bound |
| depth-p30-s08-l08 | 4 | 128 | 1024.0 | 128.0 | 0.5 | 128.0 | 0.223971 | arithmetic_lower_bound |
| depth-p30-s12-l12 | 4 | 128 | 1024.0 | 128.0 | 2.0 | 128.0 | 0.416048 | arithmetic_lower_bound |
| depth-p30-s16-l08 | 4 | 128 | 4224.0 | 128.0 | 1.0 | 128.0 | 0.190634 | arithmetic_lower_bound |
| depth-p30-s16-l16 | 4 | 128 | 1664.0 | 128.0 | 2.0 | 128.0 | 0.348453 | arithmetic_lower_bound |
| depth-p30-s24-l12 | 4 | 128 | 1536.0 | 128.0 | 1.0 | 128.0 | 0.310554 | arithmetic_lower_bound |
| depth-p30-s24-l24 | 4 | 128 | 512.0 | 128.0 | 1.0 | 128.0 | 0.487157 | arithmetic_lower_bound |
| depth-p30-s32-l08 | 4 | 128 | 3584.0 | 128.0 | 2.0 | 128.0 | 0.290763 | arithmetic_lower_bound |
| depth-p30-s32-l16 | 4 | 128 | 1664.0 | 128.0 | 5.0 | 128.0 | 0.378562 | arithmetic_lower_bound |
| depth-p30-s36-l12 | 4 | 128 | 1792.0 | 128.0 | 1.0 | 128.0 | 1.75459 | arithmetic_lower_bound |
| depth-p30-s40-l08 | 4 | 128 | 3072.0 | 128.0 | 0.5 | 128.0 | 0.288682 | arithmetic_lower_bound |
| depth-p50-s00-l08 | 4 | 128 | 1664.0 | 128.0 | 0.5 | 128.0 | 0.429882 | arithmetic_lower_bound |
| depth-p50-s00-l12 | 4 | 128 | 512.0 | 128.0 | 1.0 | 128.0 | 0.124902 | arithmetic_lower_bound |
| depth-p50-s00-l16 | 4 | 128 | 1024.0 | 128.0 | 0.5 | 128.0 | 0.288663 | arithmetic_lower_bound |
| depth-p50-s00-l24 | 4 | 128 | 512.0 | 128.0 | 0.5 | 128.0 | 0.420971 | arithmetic_lower_bound |
| depth-p50-s08-l08 | 4 | 128 | 2688.0 | 128.0 | 0.5 | 128.0 | 0.470591 | arithmetic_lower_bound |
| depth-p50-s12-l12 | 4 | 128 | 1152.0 | 128.0 | 1.0 | 128.0 | 0.115165 | arithmetic_lower_bound |
| depth-p50-s16-l08 | 4 | 128 | 4736.0 | 128.0 | 1.0 | 128.0 | 0.245676 | arithmetic_lower_bound |
| depth-p50-s16-l16 | 4 | 128 | 1024.0 | 128.0 | 1.0 | 128.0 | 0.296577 | arithmetic_lower_bound |
| depth-p50-s24-l08 | 4 | 128 | 1536.0 | 128.0 | 1.0 | 128.0 | 0.319266 | arithmetic_lower_bound |
| depth-p50-s24-l12 | 4 | 128 | 1280.0 | 128.0 | 1.0 | 128.0 | 0.268608 | arithmetic_lower_bound |
| depth-p50-s24-l24 | 4 | 128 | 1920.0 | 128.0 | 2.0 | 128.0 | 0.161998 | arithmetic_lower_bound |
| depth-p50-s32-l08 | 4 | 128 | 2048.0 | 128.0 | 1.0 | 128.0 | 0.326505 | arithmetic_lower_bound |
| depth-p50-s36-l12 | 4 | 128 | 1280.0 | 128.0 | 1.0 | 128.0 | 0.286243 | arithmetic_lower_bound |
| depth-p50-s40-l08 | 4 | 128 | 7552.0 | 128.0 | 1.0 | 128.0 | 0.181383 | arithmetic_lower_bound |
| m-sweep-p50-m12 | 12 | 0 | 256.0 | 256.0 | None | 0.0 | 2.98383 | arithmetic_lower_bound |
| m-sweep-p50-m16 | 16 | 0 | 256.0 | 640.0 | None | 0.0 | 1.30474 | arithmetic_lower_bound |
