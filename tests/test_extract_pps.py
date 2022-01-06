import csv
import os

from src.pp_head_extraction.extract_pps import write_pps_to_file


def test_print_extract_ambiguous_pp():
    in_file = "tests/data/gold_test_output.tsv"
    out_file = "tests/data/temp.tsv"

    write_pps_to_file(in_file, out_file)

    gold_file = "tests/data/gold_out_extract_pps.tsv"

    with open(gold_file) as gold, open(out_file) as out:
        gold_data = list(csv.reader(gold, delimiter="\t"))
        out_data = list(csv.reader(out, delimiter="\t"))

        print(gold_data)

    if os.path.exists(out_file):
        os.remove(out_file)

    assert len(gold_data) == len(out_data)

    for g, o in zip(gold_data, out_data):
        # Once a part of the assertion is about to fail, print it.
        # To capture print output in pytest use the flag: --capture=teesys
        if g != o:
            print(g, o)
        assert g == o
