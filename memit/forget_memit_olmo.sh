#!/bin/bash

# 循环调用程序，传递不同的次序参数
for i in {37..161}
do
    python -m experiments.evaluate_olmo --order=$i

done

#for i in 4 37 40 44 59 77 90 105 141 147
#do
#    python -m experiments.memit_jailbreak_evaluate_olmo --order=$i
#
#done
