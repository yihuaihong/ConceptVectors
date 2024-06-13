#!/bin/bash

# 循环调用程序，传递不同的次序参数
#for i in {0..94}
#do
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=5e-5 forget_loss=npo
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=5e-5 forget_loss=npo_KL
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=2e-5 forget_loss=grad_ascent
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=2e-5 forget_loss=grad_diff #对grad_diff来说，可能知识集中会更好
#    #python forget.py order=$i batch_size=8 gradient_accumulation_steps=8 gradient_checkpointing=True lr=5e-5 forget_loss=npo_KL ft_type=Niddle
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=7e-5 forget_loss=npo_KL ft_type=all_value_vectors set=test #ft_type=all_value_vectors
#    python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=7e-5 forget_loss=grad_ascent ft_type=Niddle set=test
#     #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=True lr=5e-5 forget_loss=dpo
#    python MEMIT/memit/experiments/evaluate.py --order = $i
#done
#python evaluate_llama.py

for i in 26  #18 26 27  #{0..9}
do
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=5 gradient_checkpointing=True lr=1e-5 forget_loss=npo_KL
     #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=2 gradient_checkpointing=True lr=5e-1 forget_loss=dpo ft_type=Needle set=dev
     #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=True lr=2e-1 forget_loss=dpo ft_type=Needle set=dev
     #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=True lr=1e-4 forget_loss=dpo ft_type=Needle set=test
     python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=True lr=5e-2 forget_loss=grad_diff ft_type=Needle set=test

done


#on olmo jailbreak: 4 37 40 44 59 77 90 105 141 147