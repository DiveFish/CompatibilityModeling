import csv
import sys
from dataclasses import dataclass
from typing import Dict
from typing import List


@dataclass
class PPHeadCandidate():
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


def get_instances(extr_head_file: str):
    instances = []
    instance_id = 1
    pp_id = 1

    with open(extr_head_file, "r") as headf:
        reader = csv.reader(headf, delimiter="\t")

        for line in reader:
            sent_id = int(line[0])
            pp_info = line[1:4]
            pp_noun_info = line[4:7]

            cand_info = line[7:]
            # Separate info for all head candidates (one nested list per candidate).
            candidates = [cand_info[i * 7:(i + 1) * 7] for i in range((len(cand_info) + 7 - 1) // 7)]

            for candidate in candidates:
                instance_args = [sent_id, pp_id, instance_id] + pp_info + pp_noun_info + candidate
                instances.append(PPHeadCandidate(*instance_args))
                instance_id += 1

            pp_id += 1

    return instances


def cands_per_pp(instances: List[PPHeadCandidate]):
    abs_cands = {}
    for instance in instances:
        if abs_cands.get(instance.pp_id):
            abs_cands[instance.pp_id] += 1
        else:
            abs_cands[instance.pp_id] = 1

    return abs_cands


def avg_cands(abs_cands: Dict[int, int]):
    return round(sum(abs_cands.values()) / len(abs_cands), 3)


def cands_per_topo_per_pp(instances: List[PPHeadCandidate]):
    cands_per_topo = {}
    for instance in instances:
        if not cands_per_topo.get(instance.prep_topo):
            cands_per_topo[instance.prep_topo] = {}
        if cands_per_topo[instance.prep_topo].get(instance.pp_id):
            cands_per_topo[instance.prep_topo][instance.pp_id] += 1
        else:
            cands_per_topo[instance.prep_topo][instance.pp_id] = 1

    return cands_per_topo


def avg_cands_per_topo(cands_per_topo: Dict[str, Dict[int, int]]):
    averages = {}
    for topo, instances in cands_per_topo.items():
        averages[topo] = round(sum(instances.values()) / len(instances), 3)

    return averages


def total_per_field(cands_per_topo):
    total_per_topo = {}
    for topo, instances in cands_per_topo.items():
        total_per_topo[topo] = len(instances)

    return total_per_topo


def write_tueba_stats(devf, testf, trainf):
    """
    Function only applies to the TueBa-D/Z data.
    """
    stats = {"dev": [], "test": [], "train": []}
    overall_total_pps = 0
    overall_total_cands = 0
    overall_total_pps_per_topo = {"MF": 0, "NF": 0, "VF": 0}
    overall_total_cands_per_topo = {"MF": 0, "NF": 0, "VF": 0}

    for f, split in ((devf, "dev"), (testf, "test"), (trainf, "train")):
        instances = get_instances(f)

        abs_cands = cands_per_pp(instances)
        average = avg_cands(abs_cands)
        cands_per_topo = cands_per_topo_per_pp(instances)
        total_per_topo = total_per_field(cands_per_topo)

        # Add total value and average for split.
        stats[split].append(sum(total_per_topo.values()))
        stats[split].append(average)

        # Add number of PPs and number of Candidates to overall numbers.
        overall_total_pps += sum(total_per_topo.values())
        overall_total_cands += sum(abs_cands.values())

        averages = avg_cands_per_topo(cands_per_topo)

        # Add values per field for split.
        for field in ("MF", "NF", "VF"):
            stats[split].append(total_per_topo[field])
            stats[split].append(averages[field])

            overall_total_pps_per_topo[field] += total_per_topo[field]
            overall_total_cands_per_topo[field] += sum(cands_per_topo[field].values())

    # Add line for train, test and dev combined (=overall).
    overall_output = ["overall", overall_total_pps, round(overall_total_cands / overall_total_pps, 2)]
    for field in ("MF", "NF", "VF"):
        total_pps = overall_total_pps_per_topo[field]
        average = round((overall_total_cands_per_topo[field] / overall_total_pps_per_topo[field]), 2)
        overall_output.append(total_pps)
        overall_output.append(average)

    with open("stats/tueba_stats.tsv", "w") as tueba_out:
        writer = csv.writer(tueba_out, delimiter="\t")
        writer.writerow(["split", "total_pps", "avg_heads_per_pp", "total_pps_MF", "avg_MF", "total_pps_NF", "avg_NF", "total_pps_VF", "avg_VF"])
        for split, stat in stats.items():
            writer.writerow([split] + stat)
        writer.writerow(overall_output)


def write_tueba_stats_file(train, dev, test):
    """
    Function only applies to TueBa-D/Z file.
    """
    train_pps = collect_pp_candidates(train)
    dev_pps = collect_pp_candidates(dev)
    test_pps = collect_pp_candidates(test)

    print("train\tdev\ttest")
    for i in range(len(train_pps)):
        print(train_pps[i], end="")
        if i < len(test_pps):
            if i < len(dev_pps):
                print("\t"+str(dev_pps[i])+"\t"+str(test_pps[i]), end="")
            else:
                print("\t\t"+str(test_pps[i]), end="")
        print()


def collect_pp_candidates(file):
    instances = get_instances(file)
    list_pp_cands = [value for value in cands_per_pp(instances).values()]
    return list_pp_cands


if __name__ == "__main__":

    """
    The arguments must hold the files in the following order:
    1. dev data
    2. test data
    3. train data
    """
    write_tueba_stats_file(sys.argv[1], sys.argv[2], sys.argv[3])
