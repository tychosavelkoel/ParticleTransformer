#!/bin/bash

LR=$1
WC=$2
NU=$3
DELTA=$4
EPS=$5

modelopts="networks/example_ParticleTransformer.py --use-amp --optimizer-option weight_decay 0.01"


python main.py \
    ParT ParT ../log/mnist_test ../data --objective soft-boundary\
     --lr ${LR} --lr_milestone 50 --batch_size 512 --weight_decay ${WC} \
     --nu ${NU} --warm_up_n_epochs 20\
     --data-train "/data/alice/tqwsavelkoel/soft/JetToyHI/Data_Train_klein.root" \
     --data-test "/data/alice/tqwsavelkoel/soft/JetToyHI/Data_Test_klein.root" \
     --network-config $modelopts --data-config ../data/Test/test_kin.yaml \
     --log logs/Test/Test_${model}_{auto}.log --model-prefix training/Test/{datum}/${model}/{auto}/net \
     --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
     --samples-per-epoch 80000 --samples-per-epoch-val 10000 \
     --min-epochs 5 --max-epochs 22 --gpus 0 \
     --predict-output test_results.root --optimizer ranger \
     --tensorboard Quenched_${FEATURE_TYPE}_${model} \
     --delta ${DELTA} --epsilon ${EPS}

     
