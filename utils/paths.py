from datetime import datetime
from pathlib import Path

"""
Definitions of paths to the main folders for reading and saving data and functions related to directories and file structue
"""

# PATHS THAT USUALLY DO NOT CHANGE
PROJ_DIR = Path(__file__).resolve().parent.parent
# main directory for the graph data
DATA_DIR = PROJ_DIR / Path('data')
# subdirectories of the main data directory:
ARA_DIR = DATA_DIR / Path('ara1.1')                 # directory with the action graphs (i.e. ARA corpus)
#DATA_DIR = PROJ_DIR / Path('data_ara2')
#ARA_DIR = DATA_DIR / Path('ara2.0')
SENT_AMR_DIR = DATA_DIR / Path('recipe_amrs_sentences')     # directory where the S-AMRs are located
ACTION_AMR_DIR = DATA_DIR / Path('recipe_amrs_actions')     # directory where the A-AMRs get saved
RAW_COREF_DIR = DATA_DIR / Path('coref_data_raw')           # directory with the raw coreference clusters
JOINED_COREF_DIR = DATA_DIR / Path('coref_data_joined')     # directory where joined coreference information is stored / located

# Directory for storing the logged information about e.g. non-separable AMRs
LOG_DIR = PROJ_DIR / Path('logs')


def get_new_amr_dir() -> Path:
    """
    Creates the directory for the action level AMRs if not existent
    :return: returns the path to the directory
    """
    ACTION_AMR_DIR.mkdir(exist_ok=True, parents=True)
    return ACTION_AMR_DIR


def get_new_dish_dir(recipe_name: str) -> Path:
    """
    Creates a subdirectory for a specific dish in the ACTION_AMR_DIR folder
    :param recipe_name: name of a dish
    :return: path to the directory for the dish
    """
    main_dir = get_new_amr_dir()
    dish_dir = main_dir / recipe_name
    dish_dir.mkdir(exist_ok=True, parents=True)
    return dish_dir


def get_splitting_log_path() -> Path:
    """
    Creates the file for storing counts / distributions of the splitting process
    Files are named 'splitting_log.txt' with the current date and time as prefix
    :return: Path to the log file
    """
    date_and_time = datetime.now()
    pref = f'{date_and_time.date()}-{date_and_time.time()}_'
    pref = pref.replace('.', '-')
    pref = pref.replace(':', '-')
    log_file = pref + 'splitting_log.txt'
    return LOG_DIR / log_file


def get_non_sep_log_path() -> Path:
    """
    Creates a file for logging the names of the AMRs that were not separable
    Files are named 'non_separable_amrs.txt' with the current date and time as prefix
    :return: Path to the log file
    """
    date_and_time = datetime.now()
    pref = f'{date_and_time.date()}-{date_and_time.time()}_'
    pref = pref.replace('.', '-')
    pref = pref.replace(':', '-')
    log_file = pref + 'non_separable_amrs.txt'
    with open(LOG_DIR / log_file, 'w', encoding="utf-8") as lf:
        pass
    return LOG_DIR / log_file

