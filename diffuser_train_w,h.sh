#!/bin/bash

# 첫 번째 실행: CUDA device 0 사용
OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=4 python scripts/train.py --dataset "walker2d-medium-expert-v2" --logbase logs --horizon 4 --n_diffusion_steps 2 --seed 0 --prefix 'diffusion/diffuser'   > output/we_gpu0.log 2>&1 &

# 두 번째 실행: CUDA device 1 사용
OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=5 python scripts/train.py --dataset "walker2d-medium-replay-v2" --logbase logs --horizon 4 --n_diffusion_steps 2 --seed 0 --prefix 'diffusion/diffuser'   > output/wr_gpu1.log 2>&1 &

# 세 번째 실행: CUDA device 2 사용
OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=6 python scripts/train.py --dataset "hopper-medium-expert-v2" --logbase logs --horizon 4 --n_diffusion_steps 2 --seed 0 --prefix 'diffusion/diffuser'   > output/he_gpu2.log 2>&1 &

# 네 번째 실행: CUDA device 3 사용
OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=7 python scripts/train.py --dataset "hopper-medium-replay-v2" --logbase logs --horizon 4 --n_diffusion_steps 2 --seed 0 --prefix 'diffusion/diffuser'   > output/hr_gpu3.log 2>&1 &

#OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=3 python scripts/train.py --dataset "hopper-medium-replay-v2" --logbase logs --horizon 4 --n_diffusion_steps 20 --seed 0 --prefix 'diffusion/diffuser'   > output/hr_gpu3.log 2>&1 &

#OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=3 python scripts/train.py --dataset "halfcheetah-medium-replay-v2" --logbase logs --horizon 4 --n_diffusion_steps 20 --seed 0 --prefix 'diffusion/diffuser'   > output/hr_gpu3.log 2>&1 &

#OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=3 python scripts/train.py --dataset "hopper-medium-replay-v2" --logbase logs --horizon 4 --n_diffusion_steps 20 --seed 0 --prefix 'flowmatching/flowmatcher'