from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_DIR / Path('data')
ARA_DIR = DATA_DIR / Path('ara1.1')
SENT_AMR_DIR = DATA_DIR / Path('recipe_amrs_sentences')
ACTION_AMR_DIR = DATA_DIR / Path('recipe_amrs_actions')

AMR_DIR = PROJ_DIR / Path('amr-processing')
CYCLIC_AMR_DIR = AMR_DIR / Path('parsing/aligned_cyclic_recipe_amrs')


def get_new_amr_dir():
    ACTION_AMR_DIR.mkdir(exist_ok=True, parents=True)
    return ACTION_AMR_DIR


def get_new_dish_dir(recipe_name):
    main_dir = get_new_amr_dir()
    dish_dir = main_dir / recipe_name
    dish_dir.mkdir(exist_ok=True, parents=True)
    return dish_dir
