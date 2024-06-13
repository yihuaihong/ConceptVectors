#!/bin/bash

# 循环调用程序，传递不同的次序参数
#for i in {125..161}
#do
#    python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=False lr=3e-5 forget_loss=npo_KL ft_type=Full
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=False lr=3e-5 forget_loss=npo_KL ft_type=all_value_vectors
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=False lr=2e-5 forget_loss=npo_KL ft_type=Niddle
#    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 gradient_checkpointing=False lr=3e-5 forget_loss=dpo ft_type=Full
#done
#python evaluate_olmo.py

for i in 4 37 40 44 59 77 90 105 141 147
do
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=5 gradient_checkpointing=False lr=1e-5 forget_loss=npo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=5 gradient_checkpointing=False lr=1e-5 forget_loss=dpo
    python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=False lr=6e-6 forget_loss=dpo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=8 gradient_checkpointing=False lr=4e-5 forget_loss=grad_diff
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo

    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo
    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo

    #python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=10 gradient_checkpointing=False lr=4e-5 forget_loss=dpo

done