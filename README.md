# recipe-generation

This repository contains the code for the master thesis project of generating new recipe texts based on information from two recipes for the same dish. <br>
The code for training the generation model by fine-tuning t5 for this task can be found in the [recipe-generation-model](https://github.com/interactive-cookbook/recipe-generation-model) repository.<br>
The [Wiki](https://github.com/interactive-cookbook/recipe-generation/wiki) contains more details about the algorithms implemented and AMR graphs structures and file format.

The work is still in progress. 
The currently implement steps of the planned pipeline are
* parsing each recipe sentence by sentence into AMR graphs
* separating the AMR graphs into sub-graphs in order to get one AMR per action in the corresponding action graph for the recipe 
* extracting approximated gold instructions for the split action-level amr graphs
* creating multi-sentence amr files from the amr 3.0 corpus, such thath each file contains all amrs from one document
* generating a recipe text based on an action graph and the amr graphs corresponding to each action node

## Requirements 

Tested with Python 3.6 and 3.7 and the library versions listed in brackets below. <br>
Newer Python versions should also work except possibly for the used AMR parser which was tested with Python 3.6-3.7 and which I did not test with a newer version.<br>
Note: If you do not need to run the AMR parser which requires further dependencies than the ones listed in the current section (see [amr_parsing Readme](https://github.com/interactive-cookbook/recipe-generation/tree/main/amr_parsing)) then you can delete the amr_parsing folder or simply ignore the import error warnings when opening the repository e.g. as a PyCharm project. 

The [networkX library](https://networkx.org/documentation/stable/index.html): (2.5.1; 2.6.3)
* `pip install networkx[default]`
* `pip install graphviz`

The [penman library](https://github.com/goodmami/penman/): (1.0.0; 1.2.2)
* `pip install penman`

The [pytorch library](https://pytorch.org/get-started/locally/) (1.10.1)

[Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda) from Huggingface (version 3 will probably not work): (4.11.3)
* `conda install -c huggingface transformers` (was successful)

[Sentence Piece](https://github.com/google/sentencepiece#installation) (0.1.97)
* `pip install sentencepiece`

[Spacy](https://spacy.io/usage/models) library and model for english (3.4.3):
* `pip install spacy`
* `python -m spacy download en_core_web_sm`

Other libraries:<br>
* nltk
* bs4 (only for reading the multisentence-amr xml files)

## AMR parsing 

As an intermediate representation of the recipe texts, I use Abstract Meaning Representations (AMR). 

See the [amr_parsing Readme](https://github.com/interactive-cookbook/recipe-generation/tree/main/amr_parsing) for more details on creating the AMR representations of a recipe corpus and requirements. 

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

## Generating Recipe Texts



## Creating Joined Coref Files

Information about coreference clusters, the corresponding AMR nodes and coreferences arising from the AMR splitting can be obtained by running the [coref_processing/create_joined_coref.py](https://github.com/interactive-cookbook/recipe-generation/blob/main/coref_processing/create_joined_coref.py) script.

This requires another subfolder of the data folder as described above which contains one subfolder per dish with the .jsonlines coref files. 

The paths to the action-level AMR graphs and to the coreference files is specified in utils/paths.py (ACTION_AMR_DIR and RAW_COREF_DIR). Also the path to the output folder that gets created and will contain the generated files is specified in the paths.py script (JOINED_COREF_DIR). 

Details about the output format and information included can be found at the top of the [script](https://github.com/interactive-cookbook/recipe-generation/blob/main/coref_processing/create_joined_coref.py) itself. 
