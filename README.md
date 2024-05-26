# ConceptMap Benchmark and Baselines Testing

This repository contains the data for the ConceptMap Benchmark and the code for the experiments in the paper "Blast from the past: utilizing parametric knowledge traces to evaluate unlearning". 



### Abstract
....


### Usage

1. **changing forget.yaml**

Replace the model_path and data_path with yours. And set the baseline to be tested.

Set Niddle_used to False if you don't want to use the targeted finetuning.

2. **python forget.py** 

**NEW:** 

chmod +x all_forget_llama.sh
./all_forget_llama.sh

or

chmod +x all_forget_olmo.sh
./all_forget_olmo.sh


To evaluate the baselines performance on ConceptMap Benchmark.

3. **python evaluate.py**

Evaluating the unlearned model's ability with the original ones, using both parametric metrics and QA metrics.