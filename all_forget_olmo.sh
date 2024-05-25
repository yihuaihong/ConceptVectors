#!/bin/bash

# 循环调用程序，传递不同的次序参数
for i in {0..162}
do
    python forget.py order=$i
done
