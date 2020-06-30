#!/bin/bash


dev=$1

init="h_hhhh_hhhh_hhhh"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd

eval $cmd
echo "Run $init is complete"

init="i_iiii_iiii_iiii"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd

eval $cmd
echo "Run $init is complete"

init="1_iiii_iiii_iiii"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd

eval $cmd
echo "Run $init is complete"



init="1_iiii_11ii_iiii"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd
eval $cmd
echo "Run $init is complete"



init="1_iiii_1111_iiii"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd
eval $cmd
echo "Run $init is complete"


init="1_ii11_ii11_ii11"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd
eval $cmd
echo "Run $init is complete"


init="1_1111_1111_1111"

cmd="python train.py --layers 16 --widen-factor 10  --batchnorm --lr 0.03 --init $init -d $1 --no-saves -a 'leakynet' --lrelu 0.01"
echo $cmd
eval $cmd
echo "Run $init is complete"
