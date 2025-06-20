#!/bin/bash

cd ..

# 遇到错误立即退出
set -e

# 定义手指索引列表
finger_indices="0 1 2"
# 定义手部类型列表
hand_types="gx11 gx11super gx11ball"

# 外层循环遍历手部类型
for hand_type in $hand_types
do
    echo "开始处理手部类型: $hand_type"
    # 内层循环遍历手指索引
    for finger_idx in $finger_indices
    do
        echo "  开始处理手指索引: $finger_idx"
        python ident.py --finger_idx "$finger_idx" --hand_type "$hand_type"
        echo "  完成处理手指索引: $finger_idx"
    done
    echo "完成处理手部类型: $hand_type"
done