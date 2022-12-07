from setuptools import setup

with open('requirements.txt', 'r', encoding='utf-8') as r:
    requirements = [line.strip() for line in r]

setup(name='recipe_generation',
      packages=['graph_processing',
                'amr_processing',
                'coref_processing',
                'utils'],
      install_requires=requirements
      )
