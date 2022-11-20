import os
from typing import List
from nltk.tokenize import word_tokenize
import evaluate
from statistics import mean
import argparse


# amrlib uses nltk.translated.bleu_score.corpus_bleu
# Bevilacqua et al and Ribeiro et al use sacrebleu

"""
Important:
BLEU, SacreBLEU and chrF expect as reference argument a list with one sublist per prediction,
each containing potentially several references for that specific prediction
e.g. preds = [p1, p2] and input_refs = [[ref1a, ref1b], [ref2a]] 
if instance 1 has two references and isntance 2 has only one -> number of references can be different

BLEURT expects as reference argument a list with one reference string for each prediction. 

ROUGE and METEOR can deal with both, a list with one reference string per prediction or a list of lists of reference 


ROUGE and METEOR expects that for each input string, all tokens of the sentence are separated by white spaces
BLEU and SacreBLEU use tokenizer_13a as default tokenizer, 
"""


def read_file_for_eval(file_name) -> List[str]:
    """
    Read in a file with predicted or reference sentences,
    one sentence per line
    :param file_name:
    :return: list of the sentences of the complete text
    """
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line:
                sentences.append(line)
    return sentences


def run_eval(model_output_file):
    """
    Evaluates the generated text in the model_output_file against the text in the corresponding references file
    e.g. if the file name is 'model_output.txt' then the references should be in 'model_output_reference.txt'
    Computes BLEU, Sacrebleu, chrF, ROUGE and BLEURT from https://huggingface.co/evaluate-metric
    :param model_output_file: path to a file with the model output
    :return:
    """
    generated_file = model_output_file
    gold_file = model_output_file[:-4] + '_reference.txt'

    generated_snts = read_file_for_eval(generated_file)
    ref_snts = read_file_for_eval(gold_file)

    assert len(generated_snts) == len(ref_snts)


    print(f'----- Evaluating {model_output_file} -----')
    compute_bleu(predictions=generated_snts, references=ref_snts)
    compute_chrf(predictions=generated_snts, references=ref_snts)
    compute_chrf_plus(predictions=generated_snts, references=ref_snts)
    compute_rouge(predictions=generated_snts, references=ref_snts)
    compute_meteor(predictions=generated_snts, references=ref_snts)
    compute_bleurt(predictions=generated_snts, references=ref_snts)


def compute_bleu(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/sacrebleu
    references = [[r] for r in references]
    sacre_bleu_scorer = evaluate.load("sacrebleu")
    sacrebleu = sacre_bleu_scorer.compute(predictions=predictions, references=references)
    print(f'BLEU Score: {sacrebleu}')
    return sacrebleu


def compute_chrf(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/chrf
    references = [[r] for r in references]
    chrf_scorer = evaluate.load("chrf")
    chrf = chrf_scorer.compute(predictions=predictions, references=references)
    print(f'chrF Sacre Score: {chrf}')
    return chrf


def compute_chrf_plus(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/chrf
    references = [[r] for r in references]
    chrf_scorer = evaluate.load("chrf")
    chrf_plus = chrf_scorer.compute(predictions=predictions, references=references, word_order=2)
    print(f'chrF++ Sacre Score: {chrf_plus}')
    return chrf_plus


def compute_rouge(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/rouge
    predictions = [' '.join(word_tokenize(s)) for s in predictions]
    references = [' '.join(word_tokenize(r)) for r in references]
    rouge_scorer = evaluate.load("rouge")
    rouge = rouge_scorer.compute(predictions=predictions, references=references)
    print(f'ROUGE score: {rouge}')
    return rouge


def compute_meteor(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/meteor
    predictions = [' '.join(word_tokenize(s)) for s in predictions]
    references = [' '.join(word_tokenize(r)) for r in references]
    meteor_scorer = evaluate.load("meteor")
    meteor = meteor_scorer.compute(predictions=predictions, references=references)
    print(f'METEOR score: {meteor}')
    return meteor


def compute_bleurt(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/bleurt
    bleurt_scorer = evaluate.load("bleurt", "bleurt-large-512", module_type="metric")
    bleurt = bleurt_scorer.compute(predictions=predictions, references=references)
    bleurt['average'] = mean(bleurt['scores'])
    print(f'BLEURT score: {bleurt}')
    return bleurt


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    run_eval(args.input)
    #run_eval("./output/t5_amrlib/output_amrlib_amr3_0_test.txt")
    #run_eval("./output/t5_amr/0_context/output_amr3_0_beam_1.txt")
    #run_eval("./output/t5_amr/0_context/output_ara1_split_beam_1_na.txt")



