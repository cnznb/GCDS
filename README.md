# Introduction
This document is included in the 'GECS: Detecting Extract Class Refactoring Opportunities via Deep Class Property Enhanced Graph Neural Network' distribution, which we will refer to as GECS. This is to distinguish the recommended implementation of this Extract Class refactoring from other implementations.https://anonymous.4open.science/r/REMS-A23C In this document, the environment required to make and use the GECS tool is described. Some hints about the installation environment are here, but users need to find complete instructions from other sources. They give a more detailed description of their tools and instructions for using them. Our main environment is located on a computer with windows (windows 10) operating system. The fundamentals should be similar for other platforms, although the way in which the environment is configured will be different. What do I mean by environment? For example, to run python code you will need to install a python interpreter, and if you want to use pre-trained model you will need torch.

# GECS
/src: The code files which is involved in the experiment \
/dataset: Graph Representation \
/data_demo: relevant data of the example involved in Section 3 of the paper \
/RQ3: the questionnaire and case study results \
/sampled_methods: sampled extracted classes from our collected dataset \
/tool:  a Visual Studio Code (VSCode) extension of gecs 

# Technique
## pre-trained model
CodeBERT GraphCodeBERT CodeGPT CodeT5 CoTexT PLBART

# Requirement
## CodeBERT, GraphCodeBERT, CodeGPT, CodeT5, CoTexT, PLBART
python3(>=3.6) \
we use python 3.9\
torch transformers \
we use torch(1.12.0) and transformers(4.20.1)\
pre-trained model link: \
CodeBERT: https://huggingface.co/microsoft/codebert-base \
CodeGPT: https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2 \
GraphCodeBERT: https://huggingface.co/microsoft/graphcodebert-base \
CodeT5: https://huggingface.co/Salesforce/codet5-base-multi-sum \
CoTexT: https://huggingface.co/razent/cotext-2-cc \
PLBART: https://huggingface.co/uclanlp/plbart-base

# Quickstart

##  Training phase

> step1: We gathers class structure tree, field access graph, and method call graph to form the class property graph based on fine-grained code analysis.  

spoon: [https://github.com/INRIA/spoon](https://github.com/INRIA/spoon)

> step 2: we further employs pre-trained code model to encode all the code snippets into low-dimensional vectors and construct deep class property graph based on these embedding results.

path: src/DataPreprocessing/

> step 3: we re-balances label distributions of deep class property graphs and employ graph neural network to learn implicit refactoring patterns. Finally, GECS returns a well-trained model as basis for detecting refactoring opportunities.

path: src/Training&Testing/

##  Detecton phase

> With the fine-grained code analysis and pre-trained representation generation, the deep class property graph of a target class is first extracted to capture accurate program semantics. Next, all the collected deep class property graphs are fed into the well-trained graph neural network to obtain a set of extracted field and method candidates. Finally, for each extracted field and method, GECS verifies refactoring pre-conditions and post-conditions to ensure that the suggested candidates are applicable.

path: src/Training&Testing/well-trained model

# Datasets

train data: [Tsantalis et al's dataset](https://refactoring.encs.concordia.ca/oracle/tool-refactorings/*All%20Refactorings/TP/Extract%20Class) 

real world data: [Xerces](https://github.com/apache/xerces2-j), [GanttProjects](https://github.com/bardsoftware/ganttproject)

