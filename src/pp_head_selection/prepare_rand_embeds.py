import torch

from src.pp_head_selection.dataset import get_labels
from src.pp_head_selection.dataset import read_data


UNKNOWN = "<UNK>"


def remove_outliers(data, train=True, max_dist=None, min_dist=None, gap=5):
    """
    This removes instances, where PP and candidate are too far apart.
    This is based on the head distance value.

    The keyword arguments max_dist and min_dist can only be used for test
    data. Otherwise they do not have an effect.
    """
    if train and (max_dist or min_dist):
        raise ValueError("A predefined maximum and minimum distance can only"
                         "be used for test data.")

    outl_rmvd = []

    sorted_data = iter(sorted(data, key=lambda instance: int(instance.head_distance)))

    if train:
        outl_rmvd.append(next(sorted_data))
        for instance in sorted_data:
            current_dist = int(instance.head_distance)

            latest = int(outl_rmvd[-1].head_distance)
            
            # The second part of the condition makes sure that at least one positive
            # distance is included.
            if (abs(current_dist - latest) < gap) or (current_dist > 0 and latest < 0):
                outl_rmvd.append(instance)
            else:
                if current_dist < 0:
                    outl_rmvd = [instance]
                else:
                    break
    else:
        for instance in sorted_data:
            current_dist = int(instance.head_distance)
            if min_dist <= current_dist <= max_dist:
                outl_rmvd.append(instance)

    return outl_rmvd


def value_set(clean_data):
    """
    Outliers of head distance must have been remove from the dataset.
    """
    embed_values = set()

    for instance in clean_data:
        for arg in instance.__dict__.keys():
            if arg not in ("sent_id", "instance_id", "is_head", "dep_rel"):
                embed_values.add(instance.__dict__[arg])

    # For unknown values in the test data, add the "UNKNOWN" value.
    embed_values.add(UNKNOWN)

    return embed_values


def value_to_ix(embed_values):
    return {value: i for i, value in enumerate(embed_values)}


def to_numerical(clean_data, lookup_table):
    data_matrix = []
    train_values = lookup_table.values()

    for instance in clean_data:
        instance_vector = []
        for arg, value in instance.__dict__.items():
            if arg not in ("sent_id", "instance_id", "is_head", "dep_rel"):
                try:
                    instance_vector.append(lookup_table[value])
                # For unknown values, add the "UNKNOWN" id to the dataframe.
                except KeyError:
                    instance_vector.append(lookup_table[UNKNOWN])
        data_matrix.append(instance_vector)

    return torch.as_tensor(data_matrix)


def rand_embeds(clean_train_data, clean_dev_data):
    # Train the lookup dict for numerical values.
    values = value_set(clean_train_data)
    lookup_dict = value_to_ix(values)
    value_num = len(lookup_dict)
    train_data_matrix = to_numerical(clean_train_data, lookup_dict)
    dev_data_matrix = to_numerical(clean_dev_data, lookup_dict)

    return train_data_matrix, dev_data_matrix, value_num



if __name__ == "__main__":
    train_data = read_data(".data/train-out")
    print(len(train_data))
    print(train_data[-1].sent_id)
    print(train_data[-1].instance_id)

    clean_data = remove_outliers(train_data)
    values = value_set(clean_data)
    lookup_table = value_to_ix(values)
    data_matrix = to_numerical(clean_data, lookup_table).shape
    labels = get_labels(clean_data)

    print(labels.shape)