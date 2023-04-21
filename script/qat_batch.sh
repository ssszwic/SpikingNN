#!/bin/bash
for i in {1..32}
do
    python val_qat.py \
        --cfg /home/ssszw/Work/snn/SpikingNN/run/train/fashionMnist/t17_100e/cfg.yaml \
        --weight /home/ssszw/Work/snn/SpikingNN/run/train/fashionMnist/t17_100e/best.pt \
        --qat \
        --bits ${i}
done