import csv
import os

from pyconll.load import iter_from_file
from src.pp_head_extraction.extract_ambiguous_pp import print_ambiguous_pps
from src.pp_head_extraction.extract_ambiguous_pp import field_to_set
from src.pp_head_extraction.graph import sentence_to_graph


def test_print_extract_ambiguous_pp():
    in_file = "tests/data/extract_ambi_pp.conll"
    out_file = "tests/data/temp.tsv"

    # Use the default fields VF, MF, NF.
    fields = field_to_set("")

    output = open(out_file, "w")
    for sent_id, sentence in enumerate(iter_from_file(in_file), start=1):
        grph = sentence_to_graph(sentence)
        print_ambiguous_pps(sent_id, grph, output, lemma=False, all_pp=True, fields=fields)

    output.close()

    gold_file = "tests/data/gold_test_output.tsv"

    with open(gold_file) as gold, open(out_file) as out:
        gold_data = list(csv.reader(gold, delimiter="\t"))
        out_data = list(csv.reader(out, delimiter="\t"))

    if os.path.exists(out_file):
        os.remove(out_file)

    assert len(gold_data) == len(out_data)

    for g, o in zip(gold_data, out_data):
        # Once a part of the assertion is about to fail, print it.
        # To capture print output in pytest use the flag: --capture=teesys
        if g != o:
            print(g, o)
        assert g == o
