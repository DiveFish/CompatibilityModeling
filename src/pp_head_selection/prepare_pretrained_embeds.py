import finalfusion
import json
import numpy as np
import torch
from typing import List


class TopoFieldEncoder:
    def __init__(self):
        """
        The TopoFieldEncoder captures all topoligical fields from a
        training set and sets up a look-up dictionary for them.

        A one-hot encoding can then be created for topological fields on this basis.
        """
        self.UNKNOWN = "<UNK>"
        self.label_map = {self.UNKNOWN: 0}

    def train(self, train_instances: List):
        topo_field_set = set()
        for instance in train_instances:
            topo_field_set.add(instance.prep_topo)
            topo_field_set.add(instance.pp_noun_topo)
            topo_field_set.add(instance.pp_head_topo)

        self.label_map.update(
            {field: num for num, field in enumerate(topo_field_set, start=1)}
        )

    def encode(self, instances: List):
        prep_fields = []
        pp_noun_fields = []
        pp_head_fields = []
        for instance in instances:
            # Encode topological field of preposition.
            prep_row = [0 for num in range(len(self.label_map))]
            prep_row[self.label_map[instance.prep_topo]] = 1
            prep_fields.append(prep_row)
            # Encode topological field of PP noun.
            pp_noun_row = [0 for num in range(len(self.label_map))]
            pp_noun_row[self.label_map[instance.pp_noun_topo]] = 1
            pp_noun_fields.append(pp_noun_row)
            # Encode topological field of preposition.
            pp_head_row = [0 for num in range(len(self.label_map))]
            pp_head_row[self.label_map[instance.pp_head_topo]] = 1
            pp_head_fields.append(pp_head_row)

        return (
            torch.FloatTensor(prep_fields),
            torch.FloatTensor(pp_noun_fields),
            torch.FloatTensor(pp_head_fields),
        )


class WordEncoder:
    def __init__(
        self,
        instances: List,
        word_embed_file: str,
        split: str,
        nounclass_vector_file: str = "data/nounclass_vectors.json",
    ):
        """
        Write a class that makes it possible to configure how to generate the word vectors.
        """
        self.instances = instances
        self.word_embeds = finalfusion.load_finalfusion(word_embed_file)
        self.split = split
        self.nounclass_vector_file = nounclass_vector_file

    def encode(
        self,
        average_nouns=False,
        average_other=False,
        average_ambiguous=False,
        ambiguity_threshold=3,
    ):
        # Prepositions are always encoded the same way.
        preps = self.encode_prepositions()

        if average_nouns:
            pp_nouns, pp_heads = self.encode_average_nouns(
                average_other, average_ambiguous, ambiguity_threshold
            )
        else:
            pp_nouns, pp_heads = self.encode_individual_nouns()

        return (
            torch.FloatTensor(preps),
            torch.FloatTensor(pp_nouns),
            torch.FloatTensor(pp_heads),
        )   

    def encode_prepositions(self):
        return [self.word_embeds[instance.prep] for instance in self.instances]

    def encode_individual_nouns(self):
        pp_nouns = []
        pp_heads = []

        for instance in self.instances:
            pp_nouns.append(self.word_embeds[instance.pp_noun])
            pp_heads.append(self.word_embeds[instance.pp_head])

        return pp_nouns, pp_heads

    # Make sure that the nouns are taken from the lemma list! -> is determined by input file
    def encode_average_nouns(
        self, average_other, average_ambiguous, ambiguity_threshold=3
    ):
        """
        Encode all nouns by their average nounclass vector.
        """
        def average_multiple_cats(pp_noun_cats, nounclass_vectors):
            """
            Inner function to average over multiple categories.
            """
            # Stack and average category vectors.
            cat_embeddings = None
            for cat in pp_noun_cats:
                class_vector = np.array(nounclass_vectors[cat])
                if cat_embeddings is not None:
                    cat_embeddings = np.vstack([cat_embeddings, class_vector])
                else:
                    cat_embeddings = class_vector
            cat_average = np.mean(cat_embeddings, axis=0)
            return cat_average

        # Read categories for nouns from a file.
        noun_to_cat_file = "stats/" + self.split + "_noun_to_categories.json"
        with open(self.nounclass_vector_file) as vec_file, open(
            noun_to_cat_file
        ) as n2cfile:
            nounclass_vectors = json.load(vec_file)
            noun_to_categories = json.load(n2cfile)

        pp_nouns = []
        pp_heads = []
        for instance in self.instances:
            # Go through the algorithm for PP nouns and PP heads
            for (lemma, vector_list) in zip(
                (instance.pp_noun, instance.pp_head), (pp_nouns, pp_heads)
            ):
                try:
                    pp_noun_cats = noun_to_categories[lemma]
                except KeyError:
                    vector_list.append(self.word_embeds[lemma])
                    continue
                
                num_cats = len(pp_noun_cats)

                # Case that a noun has multiple categories.
                if num_cats > 1:
                    if average_ambiguous and num_cats > ambiguity_threshold:
                        cat_average = average_multiple_cats(
                            pp_noun_cats, nounclass_vectors
                        )
                        vector_list.append(cat_average)
                    else:
                        # If no average is used, the vector of the word must be taken.
                        vector_list.append(self.word_embeds[lemma])
                # Case that a noun has just one category.
                else:
                    cat = pp_noun_cats[0]
                    if cat == "other" and not average_other:
                        vector_list.append(self.word_embeds[lemma])
                    else:
                        vector_list.append(np.array(nounclass_vectors[cat]))    

        return pp_nouns, pp_heads


def encode_pos(instances: List, pos_embed_file: str):
    pos_embeds = finalfusion.load_finalfusion(pos_embed_file)

    prep_pos = [pos_embeds[instance.prep_pos] for instance in instances]
    pp_noun_pos = [pos_embeds[instance.pp_noun_pos] for instance in instances]
    pp_head_pos = [pos_embeds[instance.pp_head_pos] for instance in instances]

    return (
        torch.FloatTensor(prep_pos),
        torch.FloatTensor(pp_noun_pos),
        torch.FloatTensor(pp_head_pos),
    )


def distance_tensor(instances: List):
    distances = [
        [int(instance.head_distance), int(instance.distance_rank)]
        for instance in instances
    ]

    return torch.FloatTensor(distances)


def pretrained_embeds(
    train_data: List,
    dev_data: List,
    word_embed_file: str,
    pos_embed_file: str,
    average_nouns=False,
    average_other=False,
    average_ambiguous=False,
    ambiguity_threshold=3,
):
    topo_encoder = TopoFieldEncoder()
    topo_encoder.train(train_data)

    # Train data setup
    word_encoder = WordEncoder(train_data, word_embed_file, split="train")
    preps, pp_nouns, pp_heads = word_encoder.encode(
        average_nouns, average_other, average_ambiguous, ambiguity_threshold
    )
    prep_pos, pp_noun_pos, pp_head_pos = encode_pos(train_data, pos_embed_file)
    prep_topos, pp_noun_topos, pp_head_topos = topo_encoder.encode(train_data)
    distances = distance_tensor(train_data)

    train_data_matrix = torch.cat(
        [
            preps,
            pp_nouns,
            pp_heads,
            prep_pos,
            pp_noun_pos,
            pp_head_pos,
            prep_topos,
            pp_noun_topos,
            pp_head_topos,
            distances,
        ],
        dim=1,
    )

    # Dev data setup
    word_encoder = WordEncoder(dev_data, word_embed_file, split="dev")
    preps, pp_nouns, pp_heads = word_encoder.encode(
        average_nouns, average_other, average_ambiguous, ambiguity_threshold
    )
    prep_pos, pp_noun_pos, pp_head_pos = encode_pos(dev_data, pos_embed_file)
    prep_topos, pp_noun_topos, pp_head_topos = topo_encoder.encode(dev_data)
    distances = distance_tensor(dev_data)

    dev_data_matrix = torch.cat(
        [
            preps,
            pp_nouns,
            pp_heads,
            prep_pos,
            pp_noun_pos,
            pp_head_pos,
            prep_topos,
            pp_noun_topos,
            pp_head_topos,
            distances,
        ],
        dim=1,
    )

    return train_data_matrix, dev_data_matrix
