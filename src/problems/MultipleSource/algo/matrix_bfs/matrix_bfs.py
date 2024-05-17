from itertools import product
from typing import Dict, List

from pyformlang.finite_automaton.state import State
from pyformlang.finite_automaton import EpsilonNFA
from pyformlang.finite_automaton.symbol import Symbol

from pygraphblas.types import BOOL
from pygraphblas.matrix import Matrix
from pygraphblas.vector import Vector
from pygraphblas import descriptor
from pygraphblas import Accum, binaryop
import pygraphblas

from src.graph.graph import Graph
from src.problems.MultipleSource.algo.matrix_bfs.reg_automaton import RegAutomaton

class MatrixIntersectionBfs:
    """
    Implementations of graph and regular grammar intersection algorithm
    """

    def __init__(self, graph: Graph, regular_automaton: RegAutomaton):
        self.graph = graph
        self.regular_automaton = regular_automaton

    def get_inverse_matrices(self) -> Dict[str, Matrix]:
        """
        Create inverse matrices from graph and regex matrices for each symbol
        """
        inverse_matrices = {}
        for symbol in self.regular_automaton.matrices:
            if str(symbol).startswith('^'):
              inverse_matrices[str(symbol)[1:]] = self.regular_automaton.matrices[symbol].T.dup()
            else:
              inverse_matrices[str(symbol)] = self.regular_automaton.matrices[symbol].T.dup()

        for symbol in inverse_matrices:
            inverse_matrices[symbol].format = pygraphblas.lib.GxB_BY_COL
        return inverse_matrices

    def get_final_states_matrix(self) -> Matrix:
        regex_vert_count = self.regular_automaton.num_states

        final_states = self.regular_automaton.final_states
        final_states_len = len(final_states)

        return Matrix.from_lists([0] * final_states_len, final_states, [True] * final_states_len, 1, regex_vert_count)

    def get_initial_traversal_matrix(self, src_verts: List[int]) -> Matrix:
        regex_start_states = self.regular_automaton.start_states

        regex_vert_count = self.regular_automaton.num_states
        graph_vert_count = self.graph.get_number_of_vertices()

        traversal_matrix = Matrix.sparse(BOOL, regex_vert_count, graph_vert_count)
        for regex_start_state in regex_start_states:
            for src_vert in src_verts:
                traversal_matrix[regex_start_state, src_vert] = True
        return traversal_matrix

    def get_vertices(self, src_verts) -> Matrix:
        """
        Intersection implementation with synchronous breadth first traversal
        of a graph and regular grammar represented in automata
        """
        regex_vert_count = self.regular_automaton.num_states
        graph_vert_count = self.graph.get_number_of_vertices()

        graph = self.graph
        regex = self.regular_automaton.matrices

        final_states_matrix = self.get_final_states_matrix()
        inverse_matrices = self.get_inverse_matrices()

        result = Matrix.sparse(BOOL, 1, graph_vert_count)

        next_traversal_matrix = self.get_initial_traversal_matrix(src_verts)
        next_visited_matrix = next_traversal_matrix.dup()

        traversal_matrix = Matrix.sparse(BOOL, regex_vert_count, graph_vert_count)
        visited_matrix = Matrix.sparse(BOOL, regex_vert_count, graph_vert_count)


        # Algo's body
        not_empty = True
        while not_empty:
            not_empty_for_at_least_one_symbol = True

            traversal_matrix.assign_matrix(next_traversal_matrix)
            visited_matrix.assign_matrix(next_visited_matrix)
            next_traversal_matrix.clear()

            # stores found nodes for each symbol

            for symbol in regex:
                with BOOL.ANY_PAIR:
                    #if str(symbol).startswith('^'):
                    #    inverse_matrices[str(symbol)[1:]].mxm(traversal_matrix.mxm(graph[str(symbol)[1:]], desc=descriptor.T1), out=next_traversal_matrix, mask=next_traversal_matrix, desc=descriptor.C)
                    #else:
                    inverse_matrices[symbol].mxm(traversal_matrix.mxm(graph[symbol]), out=next_traversal_matrix, mask=next_traversal_matrix, desc=descriptor.C & descriptor.S)

            next_visited_matrix.assign_matrix(next_traversal_matrix, mask=next_visited_matrix, desc=descriptor.C & descriptor.S)
            if visited_matrix.iseq(next_visited_matrix):
                not_empty_for_at_least_one_symbol = False

            with Accum(binaryop.MAX_BOOL):
                final_states_matrix.mxm(next_traversal_matrix, out=result)

            not_empty = not_empty_for_at_least_one_symbol

        return result

