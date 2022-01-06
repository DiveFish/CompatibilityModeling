import csv
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch import IntTensor
from typing import Tuple


class PPHeadDataset(Dataset):
    def __init__(self, data: IntTensor, labels: IntTensor):
        """
        A dataset object that is supposed to store PP head selection data.
        :param ivectors: the features per possible head for a PP
        :param labels: stores the binary values "is head" or "is not head"
        """
        self.data = data
        self.labels = labels
        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self.n_classes = torch.unique(self.labels).size(0)

    def __getitem__(self, index: int) -> Tuple[IntTensor, int]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.n_samples


@dataclass
class PPHeadCandidate:
    """
    This class represents one instance that carries:
    - a PP (preposition and noun)
    - a potential PP head
    - distance measures from head to PP noun
    - optional for training instance:
        - the information if the instance is the correct PP-head-relation
        - the dependency relation (can be "obl", "nmod" or "_")
    """

    sent_id: int
    pp_id: int
    instance_id: int
    prep: str
    prep_pos: str
    prep_topo: str
    pp_noun: str
    pp_noun_pos: str
    pp_noun_topo: str
    pp_head: str
    pp_head_pos: str
    pp_head_topo: str
    # Head distance and distance_rank are strings because the integers they hold are
    # turned into embeddings later.
    head_distance: str
    distance_rank: str
    # These are the possible labels for training instances.
    is_head: str = None
    dep_rel: str = None


def read_data(extr_head_file: str):
    instances = []
    instance_id = 1
    pp_id = 1

    with open(extr_head_file, "r") as headf:
        reader = csv.reader(headf, delimiter="\t")
        for line in reader:

            # The sentence ID is in column 0 of the TSV file.
            sent_id = int(line[0])
            
            prep_start = 1
            prep_obj_start = 4
            cands_info_start = 7
            # The indices inside the input file are described here:
            # Preposition: 1
            # POS of Preposition: 2
            # Topological Field of Preposition: 3
            prep_info = line[prep_start:prep_obj_start]
            # Prepositional Object: 4
            # POS of Prepositional Object: 5
            # Topological Field of Prepositional Object: 6
            prep_obj_info = line[prep_obj_start:cands_info_start]

            cands_info = line[cands_info_start:-1]
            single_cand_len = 6
            # The flat list is split into nested lists of 7 elements because
            # the information for one candidate is 7 items long.
            # (pp_head, pp_head_pos, pp_head_topo, head_distance,
            # distance_rank, is_head, dep_rel)
            candidates = [
                cands_info[i * single_cand_len : (i + 1) * single_cand_len]
                for i in range(
                    (len(cands_info) + single_cand_len - 1) // single_cand_len
                )
            ]

            for candidate in candidates:
                instance_args = (
                    [sent_id, pp_id, instance_id] + prep_info + prep_obj_info + candidate
                )
                instances.append(PPHeadCandidate(*instance_args))
                instance_id += 1
            
            pp_id += 1

    return instances


def get_labels(clean_data, type="is_head"):
    return torch.as_tensor([int(instance.is_head) for instance in clean_data])