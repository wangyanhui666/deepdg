dataset='PACS'
<<<<<<< HEAD
<<<<<<< HEAD
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'DANN_RES_C' 'DANN_RES_A' 'DAAN' 'DAAN_first')
=======
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'DANN_RES_C' 'DANN_RES_A' 'DAAN' 'DANN_first')
>>>>>>> f0a7744 (add DAAN first model)
=======
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'DANN_RES_C' 'DANN_RES_A' 'DAAN' 'DAAN_first')
>>>>>>> ca343d2 (fix tensorflow bug in github)
checkpoint='/model/yanhui/DANN_PACS/DANN_24/model.pkl'
test_envs=$1
gpu_ids=0
data_dir='/data/yanhui/PACS/'
net='resnet18'
task='img_dg'
output='/output/test'
logdir='/output/logs'
alpha=$4
lr=$2
max_epoch=$3
seed=$5

i=$6
tokennum=$7
mu=0
cd DeepDG/
# DANN
python train.py --mu $mu --seed $seed --alpha $alpha --checkpoint $checkpoint --logdir $logdir --lr $lr --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --tokennum $tokennum --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10 --visual