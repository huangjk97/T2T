python train.py \
--dataset cifar10 \
--num-labeled 1000 \
--arch wideresnet \
--batch-size 64 \
--mu 2 \
--lr 0.03 \
--expand-labels \
--seed 5 \
--total-steps 50000 \
--eval-step 1000 \
--out results/tmp