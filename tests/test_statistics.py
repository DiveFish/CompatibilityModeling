import pytest

from src.statistics import get_instances
from src.statistics import cands_per_pp
from src.statistics import avg_cands
from src.statistics import cands_per_topo_per_pp
from src.statistics import avg_cands_per_topo
from src.statistics import total_per_field


@pytest.fixture
def example_instances():
    test_instance_file = "tests/data/pp_head_instance_stats.tsv"
    data = get_instances(test_instance_file)
    return data


# NOTE: All numbers that are keys in the dictionaries are PP IDs, not sentence IDs!
def test_cands_per_pp(example_instances):
    assert cands_per_pp(example_instances) == {1: 3, 2: 2, 3: 2, 4: 4}


def test_avg_cands(example_instances):
    abs_cands = cands_per_pp(example_instances)
    assert avg_cands(abs_cands) == (11/4)


def test_cands_per_topo_per_pp(example_instances):
    assert cands_per_topo_per_pp(example_instances) == {"MF": {2: 2, 3: 2}, "NF": {1: 3}, "VF": {4: 4}}


def test_avg_cands_per_topo(example_instances):
    cands_per_topo = cands_per_topo_per_pp(example_instances)
    assert avg_cands_per_topo(cands_per_topo) == {"MF": 2, "NF": 3, "VF": 4}


def test_total_per_field(example_instances):
    cands_per_topo = cands_per_topo_per_pp(example_instances)
    assert total_per_field(cands_per_topo) == {"MF": 2, "NF": 1, "VF": 1}