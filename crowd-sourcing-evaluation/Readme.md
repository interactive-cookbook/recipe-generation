## Crowd-sourcing: human evaluation

### Materials
The folder [recipe_texts](https://github.com/interactive-cookbook/recipe-generation/tree/main/crowd-sourcing-evaluation/recipe_texts) contains all the recipe texts that have been used 
in the human evaluation study, grouped into the subdirectories by condition. The recipe text in the grammaticality and coherence subfolders are the two recipes that were used as fillers.

The example_english_muffin_bread_5.txt recipe was used as the example shown to participants before beginning with the actual rating task. 

The csv files int the[experimental_lists](https://github.com/interactive-cookbook/recipe-generation/tree/main/crowd-sourcing-evaluation/experimental_lists) directory each contain 
exactly the recipes that each participant assigned to that list rated.

### Collected data
The cleaned_data_anonymized.csv file contains the collected ratings of all participants and recipes, except for the participants filtered out due to not passing our quality / 
attention checks (see paper for details)

The file contains the following columns:
* `filename`: the name of the list csv file that was assigned to the participant
* `listnumber`: ID of the assigned list
* `participant`: randomly assigned participant ID
* `grammar`, `fluency`, `verbosity`, `structure`, `success`, `overall`: the ratings on a scale from 1 to 6 for each of the evaluation criteria
* `recipeid`: name of the recipe
* `dish`: name of the dish
* `condition`: condition, i.e. in which way the recipe was generated
    * dependency: by splitting based on syntactic dependencies
    * context: generated using the fine-tuned T5 model with previous sentence as context (GART5-1)
    * no_context: generated using the fine-tuned T5 model without context (GART5-0)
    * original: (human written) recipe from the original dataset
    * coref: was not part of the paper (rough idea: recipes were extended with explicit mentions for zero-anaphoras before running the splitting and generation pipeline)
