from amrlib.alignments.faa_aligner import FAA_Aligner


def get_alignments_recipe(sentences, amrs):
    """
    Computes the token-node alignments using FAA aligner for each sentence-amr pair in the
    input lists
    Alignmnents are based on individual sentences -> enumeration of tokens starts at 0 for each sentence
    :param sentences: list of sentences
    :param amrs: list of corresponding amrs
    :return: list of the AMRs with the ISI alignment information,
             list of the alignment strings
    """
    aligner = FAA_Aligner()
    lowered_sentences = [sent.lower() for sent in sentences]
    amrs_w_alignments, alignment_strings = aligner.align_sents(lowered_sentences, amrs)

    assert len(sentences) == len(amrs_w_alignments) == len(alignment_strings)

    return amrs_w_alignments, alignment_strings