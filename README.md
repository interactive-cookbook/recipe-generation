# recipe-generation

This repository contains the code for the paper from "From Sentence to Action: Splitting AMR Graphs for Recipe Instructions", DMR 2023.<br>
It contains the scripts to separate sentence-level AMR graphs into AMR graphs for individual action events and to generate action-event level recipe instructions for the obtained AMRs. <br>
The code for training the generation model by fine-tuning a T5-based AMR-to-text model for this task can be found in the [recipe-generation-model](https://github.com/interactive-cookbook/recipe-generation-model) repository.<br>
The [Wiki](https://github.com/interactive-cookbook/recipe-generation/wiki) contains more details about the algorithms implemented and AMR graphs structures and file format.
 
The implemented steps of the overall pipeline are
1. Parsing each recipe sentence by sentence into AMR graphs with recipe-level node-to-token alignments, see [AMR Parsing](https://github.com/interactive-cookbook/recipe-generation#amr-parsing)
2. Separating the AMR graphs into sub-graphs in order to get one AMR per action-event in the corresponding action graph for the recipe, see [AMR Splitting](https://github.com/interactive-cookbook/recipe-generation#amr-splitting)
* Extracting approximated gold instructions for the split action-level amr graphs as well as extracting action-level instructions based on dependency information only, see [Extraction](https://github.com/interactive-cookbook/recipe-generation#extracting-gold-instructions)
* Generating a recipe text based on an action graph, the amr graphs corresponding to each action node and a graph traversal, see [Generating Recipe Texts](https://github.com/interactive-cookbook/recipe-generation#generating-recipe-texts)

## Requirements 

Tested with Python 3.6 and newer versions. 

Run `pip install -e .` in the main repository directory. This will enable successful import of all modules and functions within the repository. Additionally, this will already install most of the dependencies with the following two exceptions:


The [pytorch library](https://pytorch.org/get-started/locally/) (e.g. 1.10.1)

[Transformers](https://huggingface.co/docs/transformers/installation#install-with-conda) from Huggingface (version 3 will probably not work): (e.g. 4.11.3). Depending on your OS and environment set up you can use one of these two commands for the installation.
* `conda install -c huggingface transformers` 
* `pip install transformers`


Note: Running the AMR parser requires further dependencies than the ones listed in the current section (see [amr_parsing Readme](https://github.com/interactive-cookbook/recipe-generation/tree/main/amr_parsing)). 


Spacy is only needed if the spacy tagger and dependency parser should be used for the gold extracted instruction. Otherwise stanza is used as default. <br>
[Spacy](https://spacy.io/usage/models) library and model for english:
* `pip install spacy`
* `python -m spacy download en_core_web_sm`



## AMR Parsing 

See the [Readme](https://github.com/interactive-cookbook/recipe-generation/tree/main/amr_parsing) in the **amr_parsing** folder for more details on creating the AMR representations of a recipe corpus and the requirements. If a dataset of recipe AMRs with node-to-token alignments is already available, the **amr_parsing** subfolder can be excluded to avoid the need to install the dependencies for the parser. 

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

## Extracting Gold Instructions

The separated AMRs that the splitting algorithm produces still include the original sentence corresponding to the original AMR as their '::snt'' meta data. In order to extract instructions for the separated action-level AMRs navigate to training/prepare_data_sets and run the following:

`python generate_gold_action_instructions.py --sep_dir [sep_dir] --orig_dir [orig_dir] --ara_dir [ara_dir] --out_dir [out_dir] --text`<br>
* `sep_dir`: optional; path to the parent directory with the separated action-level amrs, defaults to ACTION_AMR_DIR defined in utils/paths
* `orig_dir`: optional; path to the parent directory with the original sentence-level amrs, defaults to SENT_AMR_DIR defined in utils/paths
* `ara_dir`: optional; path to the parent directory of the ara corpus, defaults to ARA_DIR defined in utils/paths
* `out_dir`: required; path for the directory where the amrs with their newly created instructions get saved to
* `--text`: optional; include if only the sentences should be saved but without the actual amr graphs

For more details about the extraction itself see the [wiki page](https://github.com/interactive-cookbook/recipe-generation/wiki/Gold-Split-Corpus).

## Generating Recipe Texts

In order to generate one action-event level recipe based on a (specific) action graph, run <br>
`python generate_recipe.py --file [action_graph_file] --cont [context_len] --order [ordering_version] --config [configuration_file] --out [output_file]`
* `action_graph_file`: path to .conllu file with the action graph of the recipe
* `context_len`: number of previously generated sentences to include as input to the generation model (should not be larger than the context_len the model was trained with)
* `ordering_version`: optional; version of the traversing function to use, default is "pf-lf-id" 
* `configuration_file`: path to .json file with the configurations for the generation
* `output_file`: optional; the path to the file where the generated texts will be saved, each tab separated column will contain the recipe from one traversal, one sentence per line, first line is a header; if not provided then the generated texts will only be printed to the command line

In order to generate all action-event level recipes of a dataset split run <br>
`python generate_data_set_split.py --split [split_file] --type [split_type] --cont [context_length] --order [ordering_version] --config [configuration_file] --out [output_directory]`
* `split_file`: file with the assignments of the recipes to the different dataset splits
* `split_type`: the name of the split to generate
* `context_len`: number of previously generated sentences to include as input to the generation model (should not be larger than the context_len the model was trained with)
* `ordering_version`: optional; version of the traversing function to use, default is "pf-lf-id" 
* `configuration_file`: path to .json file with the configurations for the generation
* `output_dir`: optional; the path to the directory where the generated texts will be saved, if not provided it will default to a folder "output" in the main project directory; for each recipe specified in `split_file` one output file will be created (named [recipe_name]_generated.txt), one instruction per line; if `ordering_version` is "all" then the output file will contain one one column per traversal (tab separated) with a header line at the top


**ordering_version** <br>
Can be "top", "ids", "pf", "pf-lf" or "pf-lf-id", see [wiki page](https://github.com/interactive-cookbook/recipe-generation/wiki/Graph-Traversals) for details of the different traversals, or can be set to "all" to generate one recipe text for each ordering.

**configuration_file** <br>
For more information about the configuration files for recipe generation see the [recipe-generation-model readme](https://github.com/interactive-cookbook/recipe-generation-model#run-prediction). For generating from an action graph, the configuration file only needs to include the "generator_args" parameter dict. <br>
The specified "model_name_or_path" / "tokenizer_name_or_path need to point to a directory of a trained T5 based amr-to-text generation model which needs to include all the files saved when running the huggingface methods to save a model and a tokenizer. 

**split_file** <br>
The path to a .tsv file with the assignment of the recipes to different splits. 

```
train    baked_ziti_7
train    garam_masala_8
val      waffles_9
test     cauliflower_mash_1
...
```

It is also possible to pass a file that was obtained by running `create_recipe2split_assignment` from the recipe-generation-model repository (see the [Reproducible Split](https://github.com/interactive-cookbook/recipe-generation-model#ara-data-set) section). The script will take care of removing the leading path and removing the "\_gold.txt" suffix. 

**split_type** <br>
Should be "train", "val" or "test" if the split file has the format shown above, but can be set to any value that occurs in the first column of the split file. Then all recipes where the value in the first column is equal to `split_type` are chosen for generation


## Creating Joined Coref Files

Information about coreference clusters, the corresponding AMR nodes and coreferences arising from the AMR splitting can be obtained by running the [coref_processing/create_joined_coref.py](https://github.com/interactive-cookbook/recipe-generation/blob/main/coref_processing/create_joined_coref.py) script.

This requires another subfolder of the data folder as described above which contains one subfolder per dish with the .jsonlines coref files. 

The paths to the action-level AMR graphs and to the coreference files is specified in utils/paths.py (ACTION_AMR_DIR and RAW_COREF_DIR). Also the path to the output folder that gets created and will contain the generated files is specified in the paths.py script (JOINED_COREF_DIR). 

Details about the output format and information included can be found at the top of the [script](https://github.com/interactive-cookbook/recipe-generation/blob/main/coref_processing/create_joined_coref.py) itself. 
