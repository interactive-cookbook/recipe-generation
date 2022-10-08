# recipe-generation

This repository contains the code for the master thesis project of generating new recipe texts based on information from two recipes for the same dish. 

The work is still in progress. 
The currently implement steps of the planned pipeline are
* parsing each recipe sentence by sentence into AMR graphs
* separating the AMR graphs into sub-graphs in order to get one AMR per action in the corresponding action graph for the recipe 


## Requirements 
The [networkX library](https://networkx.org/documentation/stable/index.html):
* `pip install networkx[default]`
* `pip install graphviz`

The [penman library](https://github.com/goodmami/penman/):
* `pip install penman`

## AMR parsing 



## AMR Splitting 

For the details on how the AMR splitting algorithm works see the [Wiki](https://github.com/interactive-cookbook/recipe-generation/wiki/AMR-Splitting).

Create a folder `data` in the main project folder. Add the folder with the ARA 1.1 corpus to the `data` folder and call it `ara1.1`. Additionally, add the folder with the parsed sentence-level AMRs (including node-token alignments matching the token IDs of the ARA corpus) and call it `recipe_amrs_sentences`. 

Instead of naming the folders as explained above, you can adapt the `ARA_DIR` and `SENT_AMR_DIR` variables in `utils/paths.py`

Folder structures should be 
```
---data
  |---ara1.1
    |---dish1
       |---recipes
          |---dish1_0.conllu
          |---dish1_1.conllu
          ...
       |---alignments.tsv
    |---dish2
    ...
  |---recipe_amrs_sentences
     |---dish1
        |---amrs
           |---dish1_0_sentences_amr.txt
           |---dish1_1_sentences_amr.txt
           ...
     |---dish2
     ...
```

Then run the `amr_splitting.py` script. It will run the AMR splitting algorithm on all AMRs in the `recipe_amrs_sentences` folder. The separated version of the corpus will be stored in the (automatically created) folder `data/recipe_amrs_actions` with one subfolder per dish, directly containing the .txt files for each recipe. 

Additionally, two logging files will be created in the (automatically created) `logs` folder. 
* non_separable_amrs.txt: lists the names of all AMRs that were not separable as well as those that were separable using the fallback cases
* splitting_log.txt: additional information about the dataset, e.g. number of AMRs before splitting, number of AMRs after splitting, ...<br>
Each log file gets the date and time, at which it was created, as a unique prefix to avoid overwriting. 
