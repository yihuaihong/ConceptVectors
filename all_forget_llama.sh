#!/bin/bash

# 循环调用程序，传递不同的次序参数

#Testing on ConceptVectors Test set of LLaMA
for i in {0..94}  #18 26 27  #{0..9}
do
     python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=True lr=2e-1 forget_loss=grad_ascent ft_type=Needle set=test
done

