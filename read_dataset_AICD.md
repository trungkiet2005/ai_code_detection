---
dataset_info:
- config_name: T1
  features:
  - name: code
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 426431151
    num_examples: 500000
  - name: validation
    num_bytes: 85141894
    num_examples: 100000
  - name: test
    num_bytes: 1591126249
    num_examples: 1108207
  download_size: 905874814
  dataset_size: 2102699294
- config_name: T2
  features:
  - name: code
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 720046877
    num_examples: 502149
  - name: validation
    num_bytes: 144346484
    num_examples: 101176
  - name: test
    num_bytes: 803429685
    num_examples: 507874
  download_size: 710089703
  dataset_size: 1667823046
- config_name: T3
  features:
  - name: code
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 1266611775
    num_examples: 900000
  - name: validation
    num_bytes: 282625303
    num_examples: 200000
  - name: test
    num_bytes: 2193188214
    num_examples: 1000000
  download_size: 1591305995
  dataset_size: 3742425292
configs:
- config_name: T1
  data_files:
  - split: train
    path: T1/train-*
  - split: validation
    path: T1/validation-*
  - split: test
    path: T1/test-*
- config_name: T2
  data_files:
  - split: train
    path: T2/train-*
  - split: validation
    path: T2/validation-*
  - split: test
    path: T2/test-*
- config_name: T3
  data_files:
  - split: train
    path: T3/train-*
  - split: validation
    path: T3/validation-*
  - split: test
    path: T3/test-*
---
