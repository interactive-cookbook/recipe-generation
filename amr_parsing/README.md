# AMR Parsing

This folder includes the code to parse the recipes into AMR graphs and save them into files, as well as for shifting the token IDs in the node-token alignments 
produced by the parser such that the enumeration continues over the sentence boundaries. 


## AMR Parser

The AMR parser used is the one from 

> Drozdov, A., Zhou, J., Florian, R., McCallum, A., Naseem, T., Kim, Y., & Astudillo, R. F. (2022). Inducing and using alignments for 
> transition-based amr parsing. arXiv. Retrieved from https://arxiv.org/abs/2205.01464 doi: 10.48550/ARXIV.2205.01464

In particular, the parser version used is the MAP parser from Drozdov et al. (2022), trained on the AMR 3.0 corpus. (See [their repository](https://github.com/IBM/transition-amr-parser#trained-checkpoints) for more details.)

## Input Format 

The 
