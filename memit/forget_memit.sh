#!/bin/bash

# 循环调用程序，传递不同的次序参数
#for i in {0..94}
#do
#    #python -m experiments.evaluate --order=$i
#    python -m experiments.evaluate --order=$i
#
#done

for i in 16 18 21 26 27 38 42 47 49 54
do
    #python -m experiments.evaluate --order=$i
    python -m experiments.memit_jailbreak_evaluate --order=$i

done