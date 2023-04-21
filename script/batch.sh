#!/bin/bash
# python train.py save ./run/train/fashionMnist/t1_100e > ./log/t1_100e.log

# create dir
if [ -f "temp" ]; then
    rm -r temp
fi
mkdir temp

for i in {16..20}
do
    sed -e 's/time_window: 1/time_window: '${i}'/' ./cfg/snn.yaml > ./temp/snn_${i}t.yaml
    python train.py --cfg ./temp/snn_${i}t.yaml --save ./run/train/fashionMnist/t${i}_100e > ./log/fashionMnist_t${i}_100e.log &
done

sleep 10s
rm -r temp

# sed -e 's/6455/66'$var'/' test.txt > test1.txt
# cat test1.txt
# rm test1.txt