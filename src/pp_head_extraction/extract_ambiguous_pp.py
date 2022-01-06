import argparse
import csv

from src.pp_head_extraction.graph import (
    AdjacentTokens,
    Direction,
    is_relation,
    sentence_to_graph,
    TopoFieldState,
)
from igraph import Graph
from pyconll.load import iter_from_file
from pyconll.unit.token import Token
from typing import List
from typing import Set
from typing import Tuple

REL_CLAUSE_RELATIONS = ("nsubj", "obj", "obl", "nmod", "xcomp")


class CompetingHead:
    def __init__(self, token: Token, is_head: bool):
        """
        A class to store a token as a competing head.
        :param token: a Token object
        :param head: Boolean to indicate that the competing head is the
        actual head of the PP.
        """
        self.token = token
        self.is_head = is_head


class TrainingInstance:
    def __init__(self, prep: Token, prep_obj: Token, candidates: List[CompetingHead]):
        """
        An instance that stores competing head candidates for PPs.
        :param prep: the preposition Token of the PP 
        :param head: the object Token of the PP (noun inside PP) 
        :candidates: a list of potential heads for the PP
        """
        self.prep = prep
        self.prep_obj = prep_obj
        self.candidates = candidates


def relevant_head_tag(tag: str) -> bool:
    """
    Check whether a token is a noun or verb, which makes it a relevant head.
    :param tag: the form or lemma tag of a token
    :return: boolean stating whether the tag belongs to a relevant head or not
    """
    tag = tag.split("-")[1]
    if tag.startswith("N") or tag.startswith("V"):
        return True
    else:
        return False


def extract_form(token: Token, lemma=False) -> str:
    """
    Extract the word form or lemma from a token
    :param token: a Token object
    :param lemma: boolean, if true, the lemma is extracted instead of the form
    :return: the form or lemma of a Token object
    """
    if lemma:
        return token.lemma
    else:
        return token.form


def field_to_set(field_str: str) -> Set[str]:
    """
    The input "field_str" is a string that contains fields separated by spaces.
    If it is specified, this mehtod extracts the fields and puts them into a set.
    :param field_str: the string containg the fields
    :return: a set of topological field names
    """
    if field_str:
        return set(field_str.split())
    else:
        return {"VF", "MF", "NF"}


def get_topo_field(token: Token, level: int = 1) -> str:
    """
    Get the last level of the topological field information.
    Topological fields with multiple levels are separated by hyphens.
    Example of a field with three levels: NF-MF-MF
    :param tag: a Token object
    :return: the last level of the topological field information
    """
    try:
        topo_field = " ".join([str(s) for s in token.feats["TopoField"]])
        if "-" in topo_field:
            return topo_field.split("-")[-level]
        return topo_field
    except KeyError:
        raise KeyError("This token has no topological field.")


def get_topo_field_length(token: Token) -> int:
    """
    Get the number of levels the topological field information of a Token
    has.
    :param tag: a Token object
    :return: the number of levels
    """
    topo_field = " ".join([str(s) for s in token.feats["TopoField"]])
    return len(topo_field.split("-"))


def resolve_verb(graph: Graph, token: Token) -> Token:
    """
    Find the main verb of a sentence to include it as a head candidate.
    :param graph: the Graph object of a sentence
    :param token: a Token object in the Graph 
    :return: the token object of the main verb 
    """
    if (
        # Case for auxiliary verbs.
        "aux" not in token.deprel
        # Case for subclauses.
        and "mark" not in token.deprel
        # Cases for relative clauses.
        and not (
            "PRELS" in token.xpos
            and any([rel in token.deprel for rel in REL_CLAUSE_RELATIONS])
        ) 
        # Case for coordination.
        and token.deprel != "cc"
    ):
        pos = token.xpos.split("-")[1]
        if pos.startswith("V"):
            return token
        else:
            raise ValueError("The token is not a verb.")
    else:
        next_token = graph.vs.find(name=token.head)["token"]
        return resolve_verb(graph, next_token)


def find_coordinated_verbs(graph: Graph, verb_token: Token) -> List[Token]:
    """
    There are cases where a PP is connected to a verb that is part of a
    coordination with one or multiple other verbs/VPs.
    In this case all coordinated verbs are potential heads, if they are in
    front of the PP.

    :param graph: the Graph object of a sentence
    :param verb_token: the verb Token object that was identified as main verb
    :return: a list of the token objects of all coordinated verbs
    """
    coordinated_verbs = []

    coordination_head = graph.vs.find(name=verb_token.head)["token"]
    head_pos = coordination_head.xpos.split("-")[1]
    if head_pos.startswith("V"):
        coordinated_verbs.append(coordination_head)
    else:
        return coordinated_verbs

    for current_id in AdjacentTokens(graph, coordination_head.id, Direction.Succeeding):
        if current_id == verb_token.id:
            return coordinated_verbs
        current = graph.vs.find(name=current_id)["token"]
        current_pos = current.xpos.split("-")[1]
        # If root is reached, no coordinated verb can be found anymore.
        if current.head == "0":
            return coordinated_verbs
        current_head = graph.vs.find(name=current.head)["token"]
        if current_head.id == coordination_head.id and current_pos.startswith("V"):
            coordinated_verbs.append(current)
    
    # If function comes to its end, still return the coord. verb list.
    return coordinated_verbs

    # If function comes to its end, still return the coord. verb list.
    return coordinated_verbs


def preceding_is_noun(graph: Graph, p_index: str) -> bool:
    """
    Check whether the token right in front of a PP is a noun or not.
    :param graph: the graph object of a sentence
    :param p_index: the index of the prepositon in the sentence
    :return: boolean stating whether the preceding is a noun or not 
    """
    try:
        prec_token_id = AdjacentTokens(graph, p_index, Direction.Preceeding).__next__()
        prec_token = graph.vs.find(name=prec_token_id)["token"]
        preceding_is_noun = prec_token.xpos.split("-")[1].startswith("N")
    except StopIteration or TypeError:
        preceding_is_noun = False

    return preceding_is_noun


def find_competition_vf(
    graph: Graph, p_index: str, head_index: str
) -> List[CompetingHead]:
    """
    Find all competing heads for a PP in the "Vorfeld".
    The following explanation of the search is taken from the paper 
    "Extracting a PP Attachment Data Set from a German Dependency Treebank Using 
    Topological Fields, De Kok et al., 2017".
    It describes how the extraction works.
    Vorfeld: While extracting from the VF, we should take two different scenarios
    into account: (1) the preposition is immediately preceded by a noun in the VF or
    (2) the preposition is not immediately preceded by a noun in the VF. In the former
    case, we only add nominal candidates in the VF that precede the preposition. In the
    latter case, nouns in the MF are added as candidates as well. The verb candidate is
    found by scanning rightward from the preposition until we find the LK. The verb
    in the LK is then used to find the main verb.
    :param graph: the graph object of a sentence
    :param p_index: the index of the prepositon in the sentence
    :param head_index: the index of the head of the PP
    :return: a list of head candidates
    """
    candidates = list()
    main_verb_found = False

    prep = graph.vs.find(name=p_index)["token"]
    prep_topo_field_len = get_topo_field_length(prep)
    prec_is_noun = preceding_is_noun(graph, p_index)

    # Add nouns in VF before preposition.
    for token_id in AdjacentTokens(graph, p_index, Direction.Preceeding):
        token = graph.vs.find(name=token_id)["token"]
        token_pos = token.xpos.split("-")[1]

        try:
            topo_field = get_topo_field(token)
            topo_field_len = get_topo_field_length(token)
        except KeyError:
            continue

        # If the field of the token is not VF, a previous main clause is entered.
        # It is not possible that a head is in a different main clause.
        # The topological state does not need to be used here for clause exiting.
        if topo_field != "VF":
            break

        if token_pos.startswith("N"):
            is_head = head_index == token.id
            candidates.append(CompetingHead(token, is_head))

    # Track the previous topological field to know when a clause is exited.
    topo_state = TopoFieldState(get_topo_field(prep))

    for token_id in AdjacentTokens(graph, p_index, Direction.Succeeding):

        token = graph.vs.find(name=token_id)["token"]
        token_pos = token.xpos.split("-")[1]
        is_head = head_index == token.id

        try:
            topo_field = get_topo_field(token)
            topo_field_len = get_topo_field_length(token)
        except KeyError:
            continue

        # If one of the unknown topological fields is found, ignore the token.
        if topo_field in ("FKOORD", "FKONJ", "KOORD", "LV", "PARORD"):
            break

        # This allows for embedded relative and coordinated clauses to be taken into account.
        if topo_field_len != prep_topo_field_len:
            # Here the topological field state must be used for clause exiting in relative clauses.
            if main_verb_found and not topo_state.equal_or_prev(topo_field):
                break
            # If it is not a conjunction, the algorithm exits the sentence.
            if get_topo_field(token, level=2) != "FKONJ":
                continue

        # Add main verb to candidates.
        if topo_field in ("LK", "RK", "VC") and not main_verb_found:
            # If no verb can be resolved, the prepositional phrase is ignored 
            # by returning an empty candidate list.
            try:
                verb_candidate = resolve_verb(graph, token)
            except ValueError:
                return []
            is_head = head_index == verb_candidate.id
            candidates.append(CompetingHead(verb_candidate, is_head))

            # Add all verbs that are coordinated with the main verb as candidates.
            if verb_candidate.deprel == "conj":
                for coord_verb in find_coordinated_verbs(graph, verb_candidate):
                    is_head = head_index == coord_verb.id
                    candidates.append(CompetingHead(coord_verb, is_head))
            main_verb_found = True

        # Add nouns in MF only if the preposition is not preceded by a noun.
        if not prec_is_noun and topo_field == "MF" and token_pos.startswith("N"):
            candidates.append(CompetingHead(token, is_head))

        # Set the current node to the topological field state.
        if token.deprel != "punct":
            topo_state.set_current(topo_field)

    return candidates


def find_competition_mf(
    graph: Graph, p_index: str, head_index: str
) -> List[CompetingHead]:
    """
    Find all competing heads for a PP in the "Mittelfeld".
    The following explanation of the search is taken from the paper 
    "Extracting a PP Attachment Data Set from a German Dependency Treebank Using 
    Topological Fields, De Kok et al., 2017".
    It describes how the extraction works.
    Mittelfeld: To find the set of candidate heads in the MF, we scan backwards from
    a preposition until we find a token that forms the LK. Every noun on this path is
    marked as a candidate head. If the clause under consideration is a main clause,
    the finite verb is in the LK. We resolve for the main verb using the LK and add
    it to the candidate set. If the clause is a subordinate clause, the LK is normally a
    complementizer, which has an attachment to the finite verb in the RK. We use this
    verb to find the main verb and add it to the candidate set.
    :param graph: the graph object of a sentence
    :param p_index: the index of the prepositon in the sentence
    :param head_index: the index of the head of the PP
    :return: a list of head candidates 
    """
    candidates = list()
    main_verb_found = False

    prep = graph.vs.find(name=p_index)["token"]
    prep_topo_field_len = get_topo_field_length(prep)

    for token_id in AdjacentTokens(graph, p_index, Direction.Preceeding):
        token = graph.vs.find(name=token_id)["token"]
        token_pos = token.xpos.split("-")[1]
        is_head = head_index == token.id

        try:
            topo_field = get_topo_field(token)
            topo_field_len = get_topo_field_length(token)
        except KeyError:
            continue

        # If an embedded subclause is entered, the tokens are ignored.
        if prep_topo_field_len != topo_field_len:
            # Subclauses including VP coordination need to still resolve the verb.
            if "cc" != token.deprel and "C" != topo_field:  
                continue
            else:
                try:
                    verb_candidate = resolve_verb(graph, token)
                except ValueError:
                    return []
                is_head = head_index == verb_candidate.id
                candidates.append(CompetingHead(verb_candidate, is_head))
                main_verb_found = True

                # Add all verbs that are coordinated with the main verb as candidates.
                if verb_candidate.deprel == "conj":
                    for coord_verb in find_coordinated_verbs(graph, verb_candidate):
                        is_head = head_index == coord_verb.id
                        candidates.append(CompetingHead(coord_verb, is_head))
                break

        if topo_field == "MF":
            if token_pos.startswith("N"):
                candidates.append(CompetingHead(token, is_head))
        # Here the clause can be exited once the main verb is found because
        # no nouns from the VF are used as candidates.
        if topo_field in ("LK", "C"):
            # If no verb can be resolved, the prepositional phrase is ignored 
            # by returning an empty candidate list.
            try:
                verb_candidate = resolve_verb(graph, token)
            except ValueError:
                return []
            is_head = head_index == verb_candidate.id
            candidates.append(CompetingHead(verb_candidate, is_head))
            main_verb_found = True

            # Add all verbs that are coordinated with the main verb as candidates.
            if verb_candidate.deprel == "conj":
                for coord_verb in find_coordinated_verbs(graph, verb_candidate):
                    is_head = head_index == coord_verb.id
                    candidates.append(CompetingHead(coord_verb, is_head))
                    
            break

    return candidates


def find_competition_nf(
    graph: Graph, p_index: str, head_index: str
) -> List[CompetingHead]:
    """
    Find all competing heads for a PP in the "Nachfeld".
    The following explanation of the search is taken from the paper 
    "Extracting a PP Attachment Data Set from a German Dependency Treebank Using 
    Topological Fields, De Kok et al., 2017".
    It describes how the extraction works.
    Nachfeld: Processing of the NF is similar to the VF: when the preposition is im-
    mediately preceded by a noun, nouns in the NF immediately preceding the prepo-
    sition are added as candidates. If the preposition is not preceded by a noun, nouns
    in the MF are added as well. To find the main verb candidate, we scan leftward
    until we find a bracket and resolve for the main verb.
    :param graph: the graph object of a sentence
    :param p_index: the index of the prepositon in the sentence
    :param head_index: the index of the head of the PP
    :return: a list of head candidates 
    """
    candidates = list()
    main_verb_found = False

    prep = graph.vs.find(name=p_index)["token"]
    prep_topo_field_len = get_topo_field_length(prep)
    prec_is_noun = preceding_is_noun(graph, p_index)

    # Track the previous topological field to know when a clause is exited.
    topo_state = TopoFieldState(get_topo_field(prep))

    for token_id in AdjacentTokens(graph, p_index, Direction.Preceeding):
        token = graph.vs.find(name=token_id)["token"]
        token_pos = token.xpos.split("-")[1]

        try:
            topo_field = get_topo_field(token)
            topo_field_len = get_topo_field_length(token)
        except KeyError:
            continue

        # If one of the unknown topological fields is found, ignore the token.
        if topo_field in ("FKOORD", "FKONJ", "KOORD", "LV", "PARORD"):
            break

        # Break when entering the next clause.
        if topo_field_len != prep_topo_field_len:
            if main_verb_found and not topo_state.equal_or_subseq(topo_field):
                break
            continue

        if topo_field == "NF" and token_pos.startswith("N"):
            is_head = head_index == token.id
            candidates.append(CompetingHead(token, is_head))

        if topo_field in ("LK", "RK", "C", "VC") and not main_verb_found:
            # If no verb can be resolved, the prepositional phrase is ignored 
            # by returning an empty candidate list.
            try:
                verb_candidate = resolve_verb(graph, token)
            except ValueError:
                return []
            is_head = head_index == verb_candidate.id
            candidates.append(CompetingHead(verb_candidate, is_head))

            # Add all verbs that are coordinated with the main verb as candidates.
            if verb_candidate.deprel == "conj":
                for coord_verb in find_coordinated_verbs(graph, verb_candidate):
                    is_head = head_index == coord_verb.id
                    candidates.append(CompetingHead(coord_verb, is_head))

            main_verb_found = True

        if topo_field == "MF" and not prec_is_noun and token_pos.startswith("N"):
            is_head = head_index == token.id
            candidates.append(CompetingHead(token, is_head))

        # Set the current node to the topological field state.
        if token.deprel != "punct":
            topo_state.set_current(topo_field)

    return candidates


def extract_ambiguous_pps(
    graph: Graph, fields: Set[str], all_pp: bool
) -> List[TrainingInstance]:
    """
    Extract all potential heads for all PPs in a sentence.
    The sentence is provided in graph form.
    :param graph: the graph object of a sentence
    :param fields: all topological fields that should be checked
    :param all_pp: boolean to indicate whether PPs with only one head 
    candidate are also included
    :return: a list of TrainingInstance objects
    """
    instances = list()
    for edge in graph.es:
        if is_relation(edge) and edge["rel"] == "case":
            # Determine the Noun inside the PP and the preposition (= Token objects)
            prep_obj = graph.vs.find(name=edge["source"])["token"]
            prep_id = edge["target"]
            preposition = graph.vs.find(name=prep_id)["token"]

            # Determine the head of the noun in the PP, which is the head of
            # the whole PP (= Token object).
            pp_head_id = prep_obj.head
            # In case the PP has a direct edge to root, do not set up candidates.
            if pp_head_id == "0":
                continue

            pp_head = graph.vs.find(name=pp_head_id)["token"]
            pp_head_pos = pp_head.xpos

            # Check that the head is a verb or a noun.
            if not relevant_head_tag(pp_head_pos):
                continue

            try:
                pp_field = get_topo_field(preposition)
            except KeyError:
                continue

            # Only PPs in the desired fields are used.
            if pp_field not in fields:
                continue

            competition = None

            if pp_field == "VF":
                competition = find_competition_vf(graph, prep_id, pp_head_id)
                if not competition:
                    continue
            if pp_field == "MF":
                competition = find_competition_mf(graph, prep_id, pp_head_id)
                if not competition:
                    continue
            if pp_field == "NF":
                competition = find_competition_nf(graph, prep_id, pp_head_id)
                if not competition:
                    continue

            if (not all_pp) and len(competition) <= 1:
                continue

            if not any(competitor.is_head for competitor in competition):
                continue

            instances.append(TrainingInstance(preposition, prep_obj, competition))

    return instances


def compute_ranks(
    prep_id: str, candidates: List[CompetingHead]
) -> List[Tuple[int, CompetingHead]]:
    """
    Compute the distance ranks for all competing heads of a PP.
    :param prep_id: the index of the preposition
    :param candidates: a list of competing heads 
    :return: a list of tuples mapping ranks to candidates
    """
    ranks_to_cands = list()

    cands_before = [cand for cand in candidates if int(cand.token.id) < int(prep_id)]
    cands_after = [cand for cand in candidates if int(cand.token.id) > int(prep_id)]

    for num, candidate in enumerate(cands_before):
        rank = -(len(cands_before) - num)
        ranks_to_cands.append((rank, candidate))

    for rank, candidate in enumerate(cands_after, start=1):
        ranks_to_cands.append((rank, candidate))

    return ranks_to_cands


def print_ambiguous_pps(
    sent_id: int,
    graph: Graph,
    out_file: str,
    lemma: bool,
    all_pp: bool,
    fields: Set[str],
):
    """
    Print the PP candidates that have been extracted.
    :param sent_id: the index of the sentence
    :param graph: the graph representation of the sentence
    :param out_file: the name of the file to which the output is written 
    :param lemma: boolean stating whether form or lemma is used
    :param all_pp: boolean to indicate whether PPs with only one head 
    :param fields: all topoligical fields that should be checked
    """
    for instance in extract_ambiguous_pps(graph, fields, all_pp):
        writer = csv.writer(out_file, delimiter="\t")

        # Todo: A keyerror exception might be thrown here but it should not
        # because all instances have been tested in extract_ambiguous_pp.
        # Keep in mind though in case an error is thrown here.
        output = [
            sent_id,
            extract_form(instance.prep, lemma=lemma).lower(),
            instance.prep.xpos,
            get_topo_field(instance.prep),
            extract_form(instance.prep_obj, lemma=lemma),
            instance.prep_obj.xpos,
            get_topo_field(instance.prep_obj),
        ]

        ranks_to_cands = compute_ranks(instance.prep.id, instance.candidates)

        for rank, candidate in ranks_to_cands:
            token = candidate.token

            output.extend(
                [
                    extract_form(token, lemma=lemma),
                    token.xpos,
                    get_topo_field(token),
                    int(token.id) - int(instance.prep.id),
                    rank,
                    int(candidate.is_head),
                    instance.prep_obj.deprel if candidate.is_head else "_",
                ]
            )

        writer.writerow(output)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("input", help="Input CoNLL file")
    argp.add_argument("output", help="Output path for the csv file of extracted PPs")
    argp.add_argument(
        "-l", "--lemma", type=bool, default=False, help="Use lemma instead of forms"
    )
    argp.add_argument(
        "-a",
        "--all_pp",
        type=bool,
        default=True,
        help="Extract all PPs, including PPs with no head competition",
    )
    argp.add_argument("-f", "--field", help="Field to extract from")

    args = argp.parse_args()

    lemma = args.lemma
    input_path = args.input
    output_path = args.output
    # input_path = "tests/data/extract_ambi_pp.conll"
    # output_path = ".scribbles/debug_out.csv"
    all_pp = args.all_pp

    fields = field_to_set(args.field)

    output = open(output_path, "w", encoding="utf8")

    for sent_id, sentence in enumerate(iter_from_file(input_path), start=1):
        grph = sentence_to_graph(sentence)
        print_ambiguous_pps(sent_id, grph, output, lemma, all_pp, fields)

    output.close()
