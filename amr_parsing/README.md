# AMR Parsing

This folder includes the code to parse the recipes into AMR graphs and save them into files, as well as for shifting the token IDs in the node-token alignments 
produced by the parser such that the enumeration continues over the sentence boundaries. 

## AMR Parser

The AMR parser used is the one from 

> Drozdov, A., Zhou, J., Florian, R., McCallum, A., Naseem, T., Kim, Y., & Astudillo, R. F. (2022). Inducing and using alignments for 
> transition-based amr parsing. arXiv. Retrieved from https://arxiv.org/abs/2205.01464 doi: 10.48550/ARXIV.2205.01464

In particular, the parser version used is the MAP parser from Drozdov et al. (2022), trained on the AMR 3.0 corpus. 

For more information about options for trained checkpoints and dependencies see [their repository](https://github.com/IBM/transition-amr-parser#trained-checkpoints).

## Parsing the recipe corpus

Adjust the `input_path` and `output_path` variables in the main function (line 106 ff.) in the `recipes2amr_ibm.py` [parsing script](https://github.com/interactive-cookbook/recipe-generation/blob/main/amr_parsing/recipes2amr_ibm.py) and run it. 

The directory specified as `input_path` needs to contain one subdirectory per dish. Each of the subdirectories should include one file per recipe and each file should have one instruction per line. 

**Note**: The script uses tokenization by white space for creating the input to the acutal parser. So if a more advanced tokenization should be used, the input files should include already tokenized instructions that are then saved with white spaces between tokens. 

## Postprocessing of the parsed AMRs

**Shift token IDs in alignment information**

The node-token alignments produced by the parser start at 0 for each first token of each individually parsed instruction. In order to shift the alignments such that they matcht the token IDs in the recipe / action graphs (i.e. start at 1 and continue counting over the sentence boundaries), adapt the `non_shifted_dir` and `output_dir` variables in the main function of the `shifted_node_token_alignments.py` script and run the script.

**Fix cycles**

Although AMRs are noncyclic graphs, the parser produces some cyclic AMRs. So the last step before being able to run use the AMR graphs in the splitting and generation steps is to remove the cycles by running the `cyclic_amrs.py` script. The paths need to be specified in the main function (`cyclic_amr_dir` and `output_dir`)
