"""
NOTE: A lot of other statistics can be retrieved from the existing data structures created below.
A lot of information is already there!
"""

import argparse
import json
import os
from collections import Counter
from pyconll.load import iter_from_file

from src.pp_head_extraction.graph import AdjacentTokens
from src.pp_head_extraction.graph import Direction
from src.pp_head_extraction.graph import sentence_to_graph
from src.pp_head_selection.dataset import read_data


def read_candidates(candidate_file):
    id_to_candidate = {}
    for candidate in read_data(candidate_file):
        if id_to_candidate.get(candidate.sent_id):
            id_to_candidate[candidate.sent_id].append(candidate)
        else:
            id_to_candidate[candidate.sent_id] = [candidate]
    return id_to_candidate


def read_categories(noun_class_dir):
    nouns_by_category = {}
    for noun_class_file in os.listdir(noun_class_dir):
        noun_class_path = os.path.join(noun_class_dir, noun_class_file)
        noun_class = noun_class_file.split(".")[0]
        with open(noun_class_path) as f:
            nouns = []
            for line in f:
                nouns.append(line.strip())
            nouns_by_category[noun_class] = nouns

    return nouns_by_category


def write_statistics(
    noun_count,
    category_counts,
    stats_out_file,
    in_mult_categories,
    total_nouns_in_mult_cats,
    total_proper_nouns,
    combination_counts,
):
    total = sum(category_counts.values())

    out_stats = {
        "total_noun_count": noun_count,
        "total_category_count": total,
        "unique_nouns_in_multiple_categories": len(in_mult_categories),
        "total_nouns_in_multiple_categories": total_nouns_in_mult_cats,
        "total_proper_nouns": total_proper_nouns,
        "category_counts": category_counts,
        "combination_counts": combination_counts,
    }
    with open(stats_out_file, "w") as out_file:
        json.dump(out_stats, out_file, ensure_ascii=False, indent=4)


def write_other_file(other, other_file):
    with open(other_file, "w") as out_file:
        for word in other:
            out_file.write(word + "\n")


def write_noun_to_categories_file(noun2cat, noun_to_cat_file):
    with open(noun_to_cat_file, "w") as noun2cat_file:
        json.dump(noun2cat, noun2cat_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tueba_file", help="Input Tueba CoNLL file")
    parser.add_argument("candidate_file", help="Input file with candidate heads")
    parser.add_argument(
        "noun_class_dir", help="Directory that contains all files with Noun classes"
    )
    parser.add_argument("stats_out_file", help="Output statistics file.")
    parser.add_argument("other_file", help="Nouns in other statistics.")
    parser.add_argument("noun_to_category_file", help="Lemma to categories.")

    args = parser.parse_args()

    # Read all candidates from a file and assign them to a sentence ID.
    id_to_candidate = read_candidates(args.candidate_file)

    # Read all nouns and assign them to their categories.
    lemma_by_category = read_categories(args.noun_class_dir)

    # Count how often each category is present in the candidate data.
    category_counts = {cat: 0 for cat in lemma_by_category.keys()}
    # If the noun belongs to no category, it is put into the "other" category.
    category_counts["other"] = 0

    # A set of all nouns belonging to no category is set up.
    other = set()
    
    # Count how many nouns are in the candidate data.
    noun_count = 0
    # Store for each noun to which categories it was assigned.
    noun_to_cats = {}
    # Count how many proper nouns are in the data.
    total_proper_nouns = 0
    # Count total number of nouns in multiple categories.
    total_nouns_in_mult_cats = 0
    # 
    all_nouns_multiple_cats = []

    # Iterate over all sentences in the Tüba file to get access to lemmas.
    for sent_id, sentence in enumerate(iter_from_file(args.tueba_file), start=1):

        # This is necessary for the dev and train set because the entries were removed from the sets.
        if "dev" in args.tueba_file and sent_id > 8930:
            sent_id += 1
        if "train" in args.tueba_file and sent_id > 18118:
            sent_id += 1

        # Store the noun candidates that belong to the PPs of the current sentence.
        # A set is used because it can often happen that multiple PPs have the same head
        # candidates. It also keeps the algorithm from counting the PP noun multiple times.
        nouns = set()

        # If the sentence has no PP, it cannot be found in the candidate dictionary.
        try:
            # Make sure every PP noun is only counted once.
            pp_nouns = []
            # Go through all candidates for all PPs in this sentence.
            for candidate in id_to_candidate[sent_id]:
                # Add the potential PP head if it's a noun or proper noun.
                # Note that the 'nouns' variable contains a set.
                if "NOUN" in candidate.pp_head_pos:
                    nouns.add(candidate.pp_head)
                if "PROPN" in candidate.pp_head_pos:
                    total_proper_nouns += 1
                    nouns.add(candidate.pp_head)
                # Add the potential PP head if it's a noun or proper noun.
                if "NOUN" in candidate.pp_noun_pos:
                    nouns.add(candidate.pp_noun)
                    pp_nouns.append(candidate.pp_noun)
                if "PROPN" in candidate.pp_noun_pos:
                    total_proper_nouns += 1
                    nouns.add(candidate.pp_noun)
                    pp_nouns.append(candidate.pp_noun)
        # If the sentence contained no PPs, the next sentence is entered.
        except KeyError:
            continue

        # The Tüba sentence is converted to a graph.
        graph = sentence_to_graph(sentence)
        # The graph is traversed from word 0 to the last word.
        for current_id in AdjacentTokens(graph, 0, Direction.Succeeding):
            # Store the current token.
            token = graph.vs.find(name=current_id)["token"]

            # Check whether the current token is in the candidate or PP nouns.
            # Action is only taken if it is one of these nouns.
            if token.form in nouns:
                # Indicate whether the noun has been assigned a category yet.
                placed = False
                # Store the assigned categories.
                assigned_cats = []
                # Go through all category lists and see if they contain the lemma of the noun.
                for cat, lemma_list in lemma_by_category.items():
                    if token.lemma in lemma_list:   
                        # Only increase the overall noun count if the noun has not been assigned yet.
                        if not placed:
                            noun_count += 1
                        if placed:
                            total_nouns_in_mult_cats += 1
                        # Increase the count for the category anytime the noun is in the category.
                        category_counts[cat] += 1
                        # Indicate that the noun has been assigned.
                        placed = True
                        assigned_cats.append(cat)
                        # Store the information that the category belongs to the noun.
                        if noun_to_cats.get(token.lemma):
                            noun_to_cats[token.lemma].append(cat)
                        else:
                            noun_to_cats[token.lemma] = [cat]
                # If more than one category was assigned, add the list to the overall list.
                if len(assigned_cats) > 1:
                    all_nouns_multiple_cats.append(tuple(assigned_cats))
                # If the noun did not belong to any category, assign it to the "other" category.
                if not placed:
                    category_counts["other"] += 1
                    other.add(token.lemma)
                    noun_count += 1
                    if noun_to_cats.get(token.lemma):
                        noun_to_cats[token.lemma].append("other")
                    else:
                        noun_to_cats[token.lemma] = ["other"]

    # Sort the category counts in descending order.
    category_counts = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))

    # Store all nouns that belong to multiple categories and make sets out of the category lists.
    in_mult_categories = {
        lemma: set(cat_list)
        for lemma, cat_list in noun_to_cats.items()
        if len(set(cat_list)) > 1
    }

    lemma_to_categories = {
        lemma: list(set(cat_list))
        for lemma, cat_list in noun_to_cats.items()
    }

    combination_counts = Counter(all_nouns_multiple_cats)

    # Sort combination counts in descending order and join categories by hyphens.
    combination_counts = dict(
        sorted(
            {
                "-".join(cat_set): count
                for cat_set, count in combination_counts.items()
            }.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # Write statistics.
    write_statistics(
        noun_count,
        category_counts,
        args.stats_out_file,
        in_mult_categories,
        total_nouns_in_mult_cats,
        total_proper_nouns,
        combination_counts,
    )

    # Write all nouns in the "other" category to a file.
    write_other_file(other, args.other_file)

    write_noun_to_categories_file(lemma_to_categories, args.noun_to_category_file)
