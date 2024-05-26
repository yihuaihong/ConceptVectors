#!/bin/bash

# 循环调用程序，传递不同的次序参数
for i in {14..41}
do
    python forget.py order=$i batch_size=4 gradient_accumulation_steps=4 gradient_checkpointing=True lr=5e-5 forget_loss=npo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=5e-5 forget_loss=npo_KL
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=2e-5 forget_loss=grad_ascent
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=2e-5 forget_loss=grad_diff
done
