import os
import pathlib

import pytest
import pyconll

from src.pp_head_extraction.graph import sentence_to_graph


@pytest.fixture
def tests_root():
    yield pathlib.PurePath(os.path.dirname(__file__))


@pytest.fixture
def sentences_conll(tests_root):
    yield pyconll.load_from_file(tests_root / "data" / "graph.conll")


@pytest.fixture
def graphs(tests_root):
    graphs = []
    for sentence in pyconll.load_from_file(tests_root / "data" / "graph.conll"):
        graphs.append(sentence_to_graph(sentence))
    yield graphs
