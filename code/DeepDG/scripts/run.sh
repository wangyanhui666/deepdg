dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'DANN_RES_C' 'DANN_RES_A')
checkpoint='../../model/model.pkl'
batch_size=1
test_envs=$1
gpu_ids=0
data_dir='../../data/PACS/'
net='resnet18'
task='img_dg'
output='../../output/test'
alpha=$4
lr=$2
max_epoch=$3
seed=$5

i=8
mu=0
# DANN
python train.py --batch_size $batch_size --mu $mu --seed $seed --alpha $alpha --checkpoint $checkpoint --lr $lr --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10


#
#python train.py --data_dir /data/yanhui/PACS --max_epoch 2 --net 'resnet18' --task 'img_dg' --output /output/yanhui/train_output/test \
#--test_envs 0 --dataset 'PACS' --algorithm 'DANN' --mldg_beta 10
## Group_DRO
#python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
#--test_envs 0 --dataset PACS --algorithm GroupDRO --groupdro_eta 1
#
## ANDMask
#python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
#--test_envs 0 --dataset PACS --algorithm ANDMask --tau 1