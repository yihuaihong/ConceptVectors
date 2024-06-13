#!/bin/bash

#Testing on ConceptVectors Test set of OLMo
for i in {0..161}
do
    python forget.py order=$i batch_size=4 gradient_accumulation_steps=8 num_epochs=1 gradient_checkpointing=False lr=2e-1 forget_loss=grad_ascent ft_type=Needle set=test
done


#olmo jailbreak: 4 37 40 44 59 77 90 105 141 147