import pytest
from src.pp_head_extraction.graph import AdjacentTokens, Direction, Edges, sentence_to_graph

ANCESTOR = {0: 1, 1: 2, 2: None, 3: 4, 4: 2, 5: 6, 6: 2, 7: 2}


def test_sentence_to_graph(sentences_conll):
    for sentence in sentences_conll:
        g = sentence_to_graph(sentence)

        # test if the number of vertices is correct
        assert g.vcount() == len(sentence)

        # test if the number of edges is correct
        assert g.ecount() == 2 * (len(sentence) - 1)

        for edge in g.es:
            if edge["edge_type"] is Edges.Relation:
                target_id = edge.target
                source_id = edge.source
                assert int(sentence[target_id].head) == source_id + 1
                assert sentence[target_id].deprel == edge["rel"]
            elif edge["edge_type"] is Edges.Precedence:
                assert edge.source == edge.target - 1


def test_adjacent_tokens(graphs):
    for g in graphs:
        vcount = g.vcount()
        adj_pre_tokens = AdjacentTokens(g, str(vcount), Direction.Preceeding)
        for i, preceding_token_id in enumerate(adj_pre_tokens):
            assert preceding_token_id == str(vcount - 1 - i)

        adj_suc_tokens = AdjacentTokens(g, 1, Direction.Succeeding)
        for i, succeeding_token_id in enumerate(adj_suc_tokens, start=1):
            assert succeeding_token_id == str(i + 1)


@pytest.mark.skip("The class must be reimplemented and the test corrected.")
def test_ancestor_tokens(graphs):
    for g in graphs:
        for i in range(0, g.vcount()):
            ancestor_tokens = AncestorTokens(g, i)
            last_id = i
            for ancestor_token_id in ancestor_tokens:
                if last_id is not None:
                    assert ancestor_token_id == ANCESTOR[last_id]
                    last_id = ancestor_token_id
