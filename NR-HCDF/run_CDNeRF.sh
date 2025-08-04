 #!/bin/bash

SCALE=(0.33 0.2 0.3 0.33 0.3 0.01 0.1 0.005)
BOUND=(4.5 3.5 3.5 3.0 3.0 3.0 2.5 3.0)
TEST=("trex" "horns" "fortress" "fern" "room" "flower" "orchids" "leaves")
ITERS=(15000 12000 10000 9000)
FEW_SHOT=(0 9 6 3)
DIFF_REG_START_ITER=(2000 1000 1000 1000)

for i_dataset in 3
do

DATASET="data/nerf_llff_data/"${TEST[i_dataset]}
WORK_BASE=("test_LLFF/test_"${TEST[i_dataset]} "test_LLFF/test_"${TEST[i_dataset]}"/few_shot9" "test_LLFF/test_"${TEST[i_dataset]}"/few_shot6" "test_LLFF/test_"${TEST[i_dataset]}"/few_shot3")
DATASET_NAME=(${TEST[i_dataset]} ${TEST[i_dataset]}" 9-views" ${TEST[i_dataset]}" 6-views" ${TEST[i_dataset]}" 3-views")

for i_shot in 1
do

### training and test on the last checkpoint
CUDA_VISIBLE_DEVICES=0 python main_nerf.py $DATASET --workspace ${WORK_BASE[i_shot]}/test_CDNeRF --fp16 --few_shot ${FEW_SHOT[i_shot]} --scale ${SCALE[i_dataset]} --bound ${BOUND[i_dataset]} --dataset_name "${DATASET_NAME[i_shot]}" --implementation_name "CD" --iters ${ITERS[i_shot]} --use_lipshitz_color --use_lipshitz_sigma

done

done
