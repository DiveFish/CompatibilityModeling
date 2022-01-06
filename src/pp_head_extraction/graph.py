import enum

from igraph import Graph
from pyconll.unit.sentence import Sentence


class Edges(enum.Enum):
    """
    2 kinds of edges:
    Relation: mark the dependency relations between nodes
    Precedence: mark the node index of the precedent node.
    """

    Relation = 1
    Precedence = 2


def sentence_to_graph(sentence: Sentence):
    """
    Convert a sentence into a graph
    :param sentence: of 'Sentence' type from pyconll.unit.sentence.Sentence
    :return: A graph object set up from a sentence.
    """
    g = Graph(directed=True)

    # Edge count starts at 1.
    for offset, token in enumerate(sentence, start=1):
        g.add_vertex(name=str(offset), token=token)

    # The second for-loop is needed because all vertices must be present
    # for the edge setup.
    for token in sentence:
        int_id = int(token.id)
        if int_id > 1:
            g.add_edge(str(int_id - 1), token.id, edge_type=Edges.Precedence)

        head = token.head
        rel = token.deprel

        if int(head) > 0:
            g.add_edge(head, token.id, edge_type=Edges.Relation, rel=rel)

    # The edges get assigned their actual source and target vertices for easy
    # lookup.
    # 1 is added because the sentence indices start at 1 but the graph indices
    # start at 0.
    g.es["source"] = [str(edge.source + 1) for edge in g.es]
    g.es["target"] = [str(edge.target + 1) for edge in g.es]

    return g


class Direction(enum.Enum):
    """
    An enumerate marks the direction of the edges.
    Preceeding: Marks the direction of the edge is from the current node to
    another one.
    Succeeding: Marks the direction of the edge is from the some other 
    node to the current one.
    """

    Preceeding = 1
    Succeeding = 2


class AdjacentTokens:
    def __init__(self, graph: Graph, index: str, direction: Direction):
        self.graph = graph
        self.current = int(index)
        self.direction = direction

    def __iter__(self):
        return self

    def __next__(self):
        """
        An iterator keep returning the adjacent node of the current one in
        a sentence.
        :return: The index of the adjacent node.
        """
        if (
            self.direction is Direction.Succeeding
            and self.current >= self.graph.vcount()
        ):
            raise StopIteration
        if self.direction is Direction.Preceeding and self.current <= 1:
            raise StopIteration

        if self.direction is Direction.Succeeding:
            next_node_id = self.current + 1
        if self.direction is Direction.Preceeding:
            next_node_id = self.current - 1

        self.current = next_node_id
        return str(next_node_id)


def is_relation(an_edge: Edges):
    """
    Check wither an edge is of Relation type or Precedence type.
    :param an_edge: The edge that need to be checked.
    :return: a boolean, if it is a Relation type, return True.
    """
    if an_edge["edge_type"] is Edges.Relation:
        return True
    elif an_edge["edge_type"] is Edges.Precedence:
        return False
    else:
        raise ValueError("Invalid edge type!")


class TopoFieldState:
    """
    A class that stores the current topological field of a sentence graph.
    It is used to check whether a new sentence level is entered.
    This is necessary e.g. to distinguish main clauses from their subclauses.
    The instance variable self.field_order stores the order of the topological fields.
    :param current: The current topological field.
    """
    def __init__(self, current: str):
        self.current = current

        self.field_order = {"VF": 1, "LK": 2, "C": 2, "MF": 3, "RK": 4, "VC": 4, "NF": 5}

    def set_current(self, field: str):
        """
        Give the current field a new value.
        :param field: a topological field name
        """
        self.current = field

    def equal_or_prev(self, field: str) -> bool:
        """
        Check whether a field is the same field or the previous field
        as the current field. The comparison is done on the basis of
        the field order.
        :param field: a topological field name
        :return: a boolean, if it is the same or previous, return True.
        """
        if (
            self.field_order[field] == self.field_order[self.current] + 1
            or self.field_order[field] == self.field_order[self.current]
        ):
            return True
        else:
            return False

    def equal_or_subseq(self, field: str) -> bool:
        """
        Check whether a field is the same field or the subsequent field
        as the current field. The comparison is done on the basis of
        the field order.
        :param field: a topological field name
        :return: a boolean, if it is the same or subsequent, return True.
        """
        if (
            self.field_order[field] == self.field_order[self.current] - 1
            or self.field_order[field] == self.field_order[self.current]
        ):
            return True
        else:
            return False
