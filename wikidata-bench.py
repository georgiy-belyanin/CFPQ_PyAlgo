#import pytest

from src.graph.graph import Graph
from cfpq_data.graphs.readwrite.csv import graph_from_csv

from pyformlang.regular_expression.regex import Regex
from src.problems.MultipleSource.algo.matrix_bfs.matrix_bfs import MatrixIntersectionBfs
from src.problems.MultipleSource.algo.matrix_bfs.intersectionold import Intersection as IntersectionOld
from src.problems.MultipleSource.algo.matrix_bfs.reg_automaton import RegAutomaton

from src.utils.useful_paths import LOCAL_CFPQ_DATA
import cProfile

#data_path = LOCAL_CFPQ_DATA.joinpath("wikidata")
#graph = Graph.from_txt(data_path.joinpath("wikidata.txt"))
data_path = LOCAL_CFPQ_DATA.joinpath("temp")
graph = Graph.from_txt(data_path.joinpath("Graphs/geospecies2.txt"))
graph.load_bool_graph()

def test(benchmark):
    with open(data_path.joinpath('Grammars/queries.txt')) as f:
        for line in f.readlines():
            starting, query = line.strip().split(' ', 1)
            grammar = RegAutomaton(Regex(query))

            intersection = MatrixIntersectionBfs(graph, grammar)
            source_verts = [int(i) for i in starting.split(',')]
            print(starting, query, intersection.get_vertices(source_verts).nvals)

test(1)
