import argparse
import finalfusion
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys


def read_noun_to_cat_file(noun_to_cat_file):
    with open(noun_to_cat_file) as f:
        noun_to_cat_dict = json.load(f)

    return noun_to_cat_dict


def read_noun_to_cat_files(train_file, test_file, dev_file):
    """
    This excludes the "other" category.
    """
    noun_to_cat_dicts = {}
    for _file, split in zip((train_file, test_file, dev_file), ("train", "test", "dev")):
        noun_to_cat_dict = read_noun_to_cat_file(_file)
        noun_to_cat_dicts[split] = noun_to_cat_dict

    return noun_to_cat_dicts


def get_other_nouns():
    """
    File names are hardcoded.
    """
    other_nouns = []
    for other_file in ("stats/train_out_of_vocab.txt", "stats/test_out_of_vocab.txt", "stats/dev_out_of_vocab.txt"):
        with open(other_file) as other:
            other_nouns.extend([noun.strip() for noun in other.readlines()])

    return other_nouns


def get_cat_to_nouns(noun_to_cat_dicts, splits=["train", "test", "dev"]):
    """
    This excludes the "other" category.
    """
    # Make sure that all given splits are valid.
    try:
        all([split in ("train", "test", "dev") for split in splits])
    except ValueError:
        print("The possible splits are 'train', 'test' or 'dev'.")
        sys.exit()

    cat_to_nouns = {}
    for split, noun_to_cat_dict in noun_to_cat_dicts.items():
        if split in splits:
            for noun, cats in noun_to_cat_dict.items():
                for cat in cats:
                    if cat_to_nouns.get(cat):
                        cat_to_nouns[cat].add(noun)
                    else:
                        cat_to_nouns[cat] = {noun}

    other_nouns = {"other": get_other_nouns()}
    cat_to_nouns = {**cat_to_nouns, **other_nouns}

    return cat_to_nouns


def write_cat_to_nouns(cat_to_nouns, cat_to_noun_file):
    cat_to_nouns = {cat: list(nouns) for cat, nouns in cat_to_nouns.items()}
    with open(cat_to_noun_file, "w") as c2nfile:
        json.dump(cat_to_nouns, c2nfile, ensure_ascii=False, indent=4)


def calc_nounclass_vectors(cat_to_nouns, word_embeds):
    
    nounclass_vectors = {}

    for cat, nouns in cat_to_nouns.items():
        noun_embeddings = None
        for noun in nouns:
            if noun_embeddings is not None:
                noun_embeddings = np.vstack([noun_embeddings, word_embeds[noun]])
            else:
                noun_embeddings = word_embeds[noun]

        nounclass_vectors[cat] = np.mean(noun_embeddings, axis=0)

    return nounclass_vectors


def write_nounclass_vectors(nounclass_vectors, nounclass_vector_file):
    nounclass_vectors = {nounclass: vector.tolist() for nounclass, vector in nounclass_vectors.items()}
    with open(nounclass_vector_file, "w") as vec_file:
        json.dump(nounclass_vectors, vec_file, ensure_ascii=False, indent=4)


def compare_other_nouns(nounclass_vectors, word_embeds):
    # Sample out of the other noun class.
    # Should be close to: [ort, ort, ort, gruppe, nahrung, mensch, mensch, mensch]
    other_nouns = ["Hennigsdorf", "Antalya", "Burnley", "NBA",  "Füttern", "Heiner", "Anne", "Kilian", "Bürgerkind", "Cannabispflanze"]

    cosine_similarities = {}
    for other_noun in other_nouns:
        other_cossim = word_embeds[other_noun].reshape(1,-1)
        cosine_similarities[other_noun] = []
        for noun_class, nounclass_vector in nounclass_vectors.items():
            cossim = cosine_similarity(other_cossim, nounclass_vector.reshape(1,-1))
            cosine_similarities[other_noun].append((noun_class, round(cossim.item(), 2)))

    return cosine_similarities


def compare_nounclass_vectors(nounclass_vectors):
    cossims = {}
    for noun_class, nounclass_vector in nounclass_vectors.items():
        for noun_class2, nounclass_vector2 in nounclass_vectors.items():
            if noun_class != noun_class2:
                cossim = cosine_similarity(nounclass_vector.reshape(1,-1), nounclass_vector2.reshape(1,-1))
                cossims[noun_class + "_" + noun_class2] = round(cossim.item(), 2)
    
    return cossims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_noun_to_cat", help="Lemma to categories from tueba train split.")
    parser.add_argument("test_noun_to_cat", help="Lemma to categories from tueba test split.")
    parser.add_argument("dev_noun_to_cat", help="Lemma to categories from tueba dev split.")
    parser.add_argument("word_embed_file", help="The file that stores the pretrained embeddings.")
    parser.add_argument("-c", "--cat_to_noun_file", type=str, help="The file that maps categories to nouns.")
    parser.add_argument("-v", "--nounclass_vector_file", type=str, help="The file that stores the average vectors for all noun classes.")

    args = parser.parse_args()

    noun_to_cat_dicts = read_noun_to_cat_files(args.train_noun_to_cat, args.test_noun_to_cat, args.dev_noun_to_cat)
    cat_to_nouns = get_cat_to_nouns(noun_to_cat_dicts)

    if args.cat_to_noun_file:
        write_cat_to_nouns(cat_to_nouns, args.cat_to_noun_file)

    # Load finalfusion embeddings.
    word_embeds = finalfusion.load_finalfusion(args.word_embed_file)

    # Other class is not in these vectors.
    nounclass_vectors = calc_nounclass_vectors(cat_to_nouns, word_embeds)

    if args.nounclass_vector_file:
        write_nounclass_vectors(nounclass_vectors, args.nounclass_vector_file)

    other_compared = compare_other_nouns(nounclass_vectors, word_embeds)
    with open("other_comparisons.json", "w") as comp_file:
        json.dump(other_compared, comp_file, ensure_ascii=False, indent=4)

    nounclasses_compared = compare_nounclass_vectors(nounclass_vectors)
    with open("nounclass_comparisons.json", "w") as comp_file:
        json.dump(nounclasses_compared, comp_file, ensure_ascii=False, indent=4)
