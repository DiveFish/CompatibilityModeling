import pytest
import torch

from src.pp_head_selection.dataset import read_data
from src.pp_head_selection.prepare_rand_embeds import get_labels
from src.pp_head_selection.prepare_rand_embeds import remove_outliers
from src.pp_head_selection.prepare_rand_embeds import to_numerical
from src.pp_head_selection.prepare_rand_embeds import value_set
from src.pp_head_selection.prepare_rand_embeds import value_to_ix


@pytest.fixture
def example_instances():
    test_instance_file = "tests/data/pp_head_instance_data.tsv"
    data = read_data(test_instance_file)
    return data


@pytest.fixture
def gap_test_instances():
    gap_instance_file = "tests/data/gap_distance_test_instances.tsv"
    gap_instances = read_data(gap_instance_file)
    return gap_instances


@pytest.fixture
def example_values(example_instances):
    values = value_set(example_instances)
    return values


@pytest.fixture
def example_value_ixs(example_values):
    value_ixs = value_to_ix(example_values)
    return value_ixs


def test_value_to_ix(example_value_ixs):
    ix_dict = {
        "NOUN-NN": 0,
        "vertrieben": 1,
        "PROPN-NE": 2,
        "terrorisiert": 3,
        "gab": 4,
        "VERB-VVPP": 5,
        "MF": 6,
        "VERB-VVFIN": 7,
        "-6": 8,
        "ADP-APPR": 9,
        "in": 10,
        "VC": 11,
        "-3": 12,
        "-2": 13,
        "-1": 14,
        "<UNK>": 15,
        "ermordet": 16,
        "-4": 17,
        "Afrika": 18,
        "gibt": 19,
        "Ziel": 20,
        "mit": 21,
        "NF": 22,
        "LK": 23,
    }

    assert len(example_value_ixs) == len(ix_dict)
    # No exact comparison of dicts can be made because ids are given arbitrarily.
    # But the keys must match.
    assert set(example_value_ixs.keys()) == set(ix_dict.keys())


def test_to_numerical(example_instances, example_value_ixs):
    numerical = to_numerical(example_instances, example_value_ixs)

    # This is what an example output looks like (fits the ix_dict above).

    # example_numerical = [
    #     [0, 16, 2, 19, 17, 2, 7, 5, 21, 14, 8],
    #     [0, 16, 2, 19, 17, 2, 4, 5, 21, 23, 14],
    #     [0, 16, 2, 19, 17, 2, 6, 5, 21, 13, 15],
    #     [9, 16, 12, 10, 11, 12, 18, 1, 22, 15, 14],
    #     [9, 16, 12, 10, 11, 12, 20, 1, 22, 8, 15],
    # ]

    # No exact comparison is possible because of the arbitrary assignment.

    # The first three numbers must be the same for the same PP.
    assert numerical[0,:3].tolist() == numerical[1,:3].tolist() == numerical[2,:3].tolist()
    assert numerical[3,:3].tolist() == numerical[4,:3].tolist()

    assert torch.max(numerical) == max(example_value_ixs.values())
    assert torch.min(numerical) == min(example_value_ixs.values())
    # There are 5 instances because the first PP has 3 candidates and the
    # second PP has 2 candidates.
    assert numerical.size()[0] == 5
    # They all have 11 features.
    assert numerical.size()[1] == 11


def test_get_labels(example_instances):
    labels = get_labels(example_instances)

    assert labels[0] == 0
    assert labels[1] == 1
    assert labels[2] == 0
    assert labels[3] == 0
    assert labels[4] == 1


def test_remove_outliers(gap_test_instances):
    # Note that train and gap are default but added here for clearer seeing of the test case.
    outliers_removed = remove_outliers(gap_test_instances, train=True, gap=5)
    clean_distances = [int(instance.head_distance) for instance in outliers_removed]

    # original distances: [-214, -165, -151, -125, -107, -104, -55, -41, -28, -21, -16, -15, -12, -9, 
    # -8, -7, -6, -5, -4, -4, -3, -2, -1, -1, 3, 3, 3, 4, 4, 7, 9, 11, 16, 17, 48, 49, 80, 83, 87, 91, 95]

    assert [-16, -15, -12, -9, -8, -7, -6, -5, -4, -4, -3, -2, -1, -1, 3, 3, 3, 4, 4, 7, 9, 11] == clean_distances


def test_remove_outliers_at_least_one_positive(gap_test_instances):
    outliers_removed_3 = remove_outliers(gap_test_instances, train=True, gap=3)
    clean_distances_3 = [int(instance.head_distance) for instance in outliers_removed_3]
    assert [-9, -8, -7, -6, -5, -4, -4, -3, -2, -1, -1, 3, 3, 3, 4, 4] == clean_distances_3


def test_remove_outliers_test_data(gap_test_instances):
    outliers_removed = remove_outliers(gap_test_instances, train=False, max_dist=5, min_dist=-5)
    clean_distances = [int(instance.head_distance) for instance in outliers_removed]
    print(clean_distances)
    assert [-5, -4, -4, -3, -2, -1, -1, 3, 3, 3, 4, 4] == clean_distances
