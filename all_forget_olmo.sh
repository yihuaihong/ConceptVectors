#!/bin/bash

# 循环调用程序，传递不同的次序参数
for i in {145..161}
do
    python forget.py order=$i batch_size=1 gradient_accumulation_steps=16 gradient_checkpointing=False lr=5e-5 forget_loss=npo
done
