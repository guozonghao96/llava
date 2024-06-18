# vqa v2
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/eval/vqav2.sh
# echo 'vqav2'

# # gqa
# CUDA_VISIBLE_DEVICES=1 bash scripts/eval/gqa.sh
# CUDA_VISIBLE_DEVICES=3,4,5,6,7 bash scripts/eval/gqa.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/gqa.sh
# echo 'gqa'

# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/vizwiz.sh
# echo 'vizwiz'

# # sqa 
# CUDA_VISIBLE_DEVICES=2 bash scripts/eval/sqa.sh
# echo 'sqa'

# # textqa
CUDA_VISIBLE_DEVICES=1 bash scripts/eval/textvqa.sh
echo 'textqa'

# CUDA_VISIBLE_DEVICES=1 bash scripts/eval/pope.sh
# echo 'pope' 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/rec.sh
# echo 'rec'

# echo 'mme' 

# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench.sh
# echo 'mmbench'

# CUDA_VISIBLE_DEVICES=1 bash scripts/eval/mmbench_cn.sh
# echo 'mmbench-cn'

# echo 'seed bench'
# 需要下载

# CUDA_VISIBLE_DEVICES=2 bash scripts/eval/llavabench.sh
# echo 'LLaVA-Bench-in-the-Wild'

# CUDA_VISIBLE_DEVICES=3 bash scripts/eval/mmvet.sh
# echo 'mm-vet'


# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh
# echo 'mme'

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/eval/seed.sh
# echo 'seed'
