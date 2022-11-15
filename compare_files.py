from amr_processing.penman_networkx_conversions import penman2networkx
from graph_processing.read_graphs import read_aligned_amr_file


def compare_sent_by_sent(path1, path2):
    file1 = open(path1, 'r', encoding='utf-8')
    file2 = open(path2, 'r', encoding='utf-8')
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    file1.close()
    file2.close()

    for l1, l2 in zip(lines1, lines2):
        if l1 != l2:
            print("old")
            print(l1)
            print("new")
            print(l2)
            print("\n")


def compare_dict(path1, path2):
    file1 = open(path1, 'r', encoding='utf-8')
    file2 = open(path2, 'r', encoding='utf-8')
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    file1.close()
    file2.close()

    dict1 = dict()
    dict2 = dict()

    for l in lines1:
        ll = l.split('\t')
        dict1[ll[0]] = ll[1]
    for l in lines2:
        ll = l.split('\t')
        dict2[ll[0]] = ll[1]

    assert len(dict1.keys()) == len(dict2.keys())
    assert set(dict1.keys()) == set(dict2.keys())

    for k, v in dict1.items():
        if v != dict2[k]:
            print(k)
            print(v)
            print(dict2[k])
            print('\n')


if __name__=='__main__':
    path1 = './training/prepare_data_sets/all_new_sentences_ara2.txt'
    path2 = './training/prepare_data_sets/all_new_sentences_ara2_simple.txt'
    #path1 = './training/prepare_data_sets/all_new_sentences_ara2.txt'
    #path2 = './training/prepare_data_sets/sentences_ara2_comp.tsv'
    #compare_dict(path1, path2)
    compare_sent_by_sent(path1, path2)

