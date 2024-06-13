# ConceptVectors Benchmark
This repository provides the code for our paper titled **[Intrinsic Evaluation of Unlearning Using Parametric Knowledge Traces]**


## Quick Links
- [ConceptVectors Benchmark](#conceptVectors-benchmark)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
  - [1. Requirements](#1-requirements)
  - [2. Training and Forgetting](#2-training-and-forgetting)
  - [3. Evaluation Forgetting Effectiveness](#3-evaluation-forgetting-effectiveness)
  - [4. Concept Validation Experiments](#4-concept-Validation-experiments)
  - [5. Jailbreaking Experiments](#4-jailbreaking-experiments)
  - [How to Cite](#how-to-cite)

## Overview
You can reproduce the experiments of our paper.

> **Abstract**
> 
> 
<p align="center">
  <img src="asset/insight.png" width="1000"></a>
  <br />
  <em>Examples of ConceptVectors Benchmark on LLaMA and OLMO.</em>
</p>

**Instance Structure Example**:

```python
  {
      "ID": "",
      "Concept": "Harry Potter",
      "Layer": 20,
      "Dim": 10513,
      "QA": ["Who is the author of the Harry Potter book series?",
            "What is the name of the first book in the Harry Potter series?"..],
      "text_completion": [{
                "First_half": "In contrast Emily Griesinger...",
                "Second_half": "his encounter with the Sorting Hat..."
            }],
      "unrelated_QA": ["When was Costa Coffee founded?",
                      "Where is Costa Coffee headquartered?...], 
      "wikipedia_content": "Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling...",
  }
```



## 1. Requirements
To install the required packages for our baselines testing on ConceptVectors, please run the following command.
```sh
conda create -n conceptvectors python=3.9.5
conda activate conceptvectors
pip install -r requirements.txt
```

## 2. Training and Forgetting


```sh
conda create -n conceptvectors python=3.9.5
conda activate conceptvectors
pip install -r requirements.txt
```


## 3. Evaluation Forgetting Effectiveness

## 4. Concept Validation Experiments


## 5. Jailbreaking Experiments



## How to Cite


