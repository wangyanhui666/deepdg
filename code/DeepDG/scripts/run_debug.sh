dataset='PACS'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'DANN_RES_C' 'DANN_RES_A')
checkpoint=''
batch_size=1
test_envs=0
gpu_ids=0
data_dir='../../data/PACS/'
net='resnet18'
task='img_dg'
output='./output/test'
logdir='./output/logs'
alpha=0.1
lr=0.005
max_epoch=5
seed=0

i=2
mu=0
# DANN
cd ../
python train.py --batch_size 32 --mu 0 --seed 0 --alpha 0.1 --logdir ./output/logs --lr 0.005 --data_dir ../../data/PACS/ --max_epoch 5 --net resnet18 --task img_dg --output ./output/test --test_envs 0 --dataset PACS --algorithm DANN --mldg_beta 10


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