#!/bin/bash/

##############################################
### Example Script to Train QCNNs with TWN ###
##############################################
# You can use the following terminal command to send the output (stdout and stderr) to both terminal and file (log.txt in this case):
# ./script.sh 2>&1 | tee log.txt


#############
### MNIST ###
#############

python main_mnist.py --optimizer SGD --momentum 0.9 --learning_rate 0.01 --weight_decay 0 --ternarize True --batch_size 128 --epochs 100 --number_workers 4 --load_checkpoint False

python main_mnist.py -o Adam -lr 0.001 -wd 0 -t False -bs 128 -e 100 -nw 4 -lc False


################
### CIFAR-10 ###
################

python main_cifar10.py --network ResNet --layers 20 --optimizer Adam --learning_rate 0.001 --weight_decay 0 --ternarize False --batch_size 128 --epochs 200 --number_workers 2 --load_checkpoint False

python main_cifar10.py -n VGG -l 16 -bn True -o Adam -lr 0.001 -wd 0 -t True -bs 128 -e 200 -nw 2 -lc False


################
### ImageNet ###
################

CUDA_VISIBLE_DEVICES=0,1 python main_imagenet.py --network VGG --layers 16 --batch_norm True --optimizer SGD --learning_rate 0.001 --weight_decay 0 --ternarize True --batch_size 64 --epochs 10 --number_workers 8 --load_checkpoint False --distributed True

CUDA_VISIBLE_DEVICES=0 python main_imagenet.py -n AlexNet -o Adam -lr 0.001 -wd 0 -t False -bs 128 -e 100 -nw 2 -lc False -d False