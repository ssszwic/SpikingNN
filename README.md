# SNN 训练推理框架
## 环境配置
``` python
pytorch==1.12.1
```
## 模型训练
``` bash
python train.py \
        --cfg ./cfg/snn.yaml \   # 模型配置文件
        --device 0 \                # 训练设备
        --save ./run/train/test     # 训练结果保存目录
```
其中模型配置文件如下：
``` yaml
# dataset #######################################
# 类别数目
nc: 10
# 数据集位置
dataset: /home/ssszw/Work/snn/Dataset

# train parameter ###############################
learning_rate: 0.001
batch_size: 100
epoch: 100

# net parameter  ################################
thresh: 0.5
lens: 0.5
decay: 0.25
time_window: 1

# input size
height: 28
width: 28
planes: 1

net:
  # [module, args] 
  [[Conv, [6, 5, 1, 0]], # [out_planes, kernel_size, stride, padding]
   [Pool2d, [2, 2, 0]], # [kernel_size, stride, padding]
   [Conv, [12, 5, 1, 0]],
   [Pool2d, [2, 2, 0]],
   [Fc,   [10]], # out_planes
  ]
```

## 模型离线量化
``` bash
python qat.py \
        --cfg ./cfg/snn.yaml \                  # 模型配置文件
        --weight ./run/train/test/best.pt \     # 模型权重
        --bits 8 \                              # 量化位宽，默认为8
        --save ./run/train/test/qat \           # 量化权重和量化阈值保存位置
        --inlayers                              # 分层量化
```
量化后生成 qat.pt 和 thresh_qat.txt 两个文件，qat.pt 中保存量化后的权重，数据格式仍为float类型，thresh_qat.txt 中保存量化后每一层的阈值

## 模型验证
``` bash
# 对于没有量化的模型
python val.py \
        --cfg ./cfg/snn.yaml \                  # 模型配置文件
        --weight ./run/train/test/best.pt \     # 模型权重
        --device 0                              # 推理设备

# 对于量化后的模型
python val.py \
        --cfg ./cfg/snn.yaml \                              # 模型配置文件
        --weight ./run/train/test/qat/qat.pt \              # 模型权重
        --device 0 \                                        # 推理设备
        --qat \                                             # 量化推理
        --thresh_qat ./run/train/test/qat/thresh_qat.txt \  # 阈值量化文件，有离线量化得到
        --bits 8                                            # 量化位宽，默认为8
```

## 权重导出
``` bash
# 对于没有量化的模型
python val.py \
        --weight ./run/train/test/best.pt \     # 模型权重
        --save ./run/train/test/weight \        # 导出后的权重保存结果
        --plot                                  # 绘制权重直方图

# 对于量化后的模型
python val.py \
        --weight ./run/train/test/qat/qat.pt \  # 模型权重
        --save ./run/train/test/weight \        # 导出后的权重保存结果
        --qat \                                 # 权重为量化后的权重，保存为int类型
        --plot                                  # 绘制权重直方图
```
