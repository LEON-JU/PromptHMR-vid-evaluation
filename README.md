## PromptHMR 
**This is an extension** for the paper (CVPR25): \
**PromptHMR: Promptable Human Mesh Recovery**  
[Yufu Wang](https://yufu-wang.github.io), [Yu Sun](https://www.yusun.work), [Priyanka Patel](https://pixelite1201.github.io), [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/), [Michael J. Black](https://ps.is.mpg.de/person/black), [Muhammed Kocabas](https://ps.is.mpg.de/person/mkocabas)\
[[Project Page](https://yufu-wang.github.io/phmr-page)]
[[Arxiv](https://arxiv.org/abs/2504.06397)]

<img src="data/teaser.jpg" width="700">


## What does this repository changes
1. Extend evaluation methods from single-frame to world-coordinate vid seqences evaluation. 
2. Use the predefined metrics in gvhmr to implement the evaluation code.
3. **By far can only be applied to EMDB dataset**.


## How to run PromptHmr-vid evaluation
```
bash scripts/eval_emdb.sh
```

## Installation
same as orinal repository

## Dataset Preparations
download `EMDB_hmr4d_support.tar.gz` from https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link 

run `tar -xzvf EMDB_hmr4d_support.tar.gz` and put `EMDB/hmr4d_support/` under `inputs` folder of PromptHMR, if this folder does not exists, create one. 