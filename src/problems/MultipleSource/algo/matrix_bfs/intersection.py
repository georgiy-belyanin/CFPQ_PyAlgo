from itertools import product
from typing import Dict

from pyformlang.finite_automaton.state import State
from pyformlang.finite_automaton import EpsilonNFA
from pyformlang.finite_automaton.symbol import Symbol

from pygraphblas.types import BOOL
from pygraphblas.matrix import Matrix
from pygraphblas.vector import Vector
from pygraphblas import descriptor
from pygraphblas import Accum, binaryop

from src.graph.graph import Graph
from src.problems.MultipleSource.algo.matrix_bfs.reg_automaton import RegAutomaton


class Intersection:
    """
    Implementations of graph and regular grammar intersection algorithm
    """

    def __init__(self, graph: Graph, regular_automaton: RegAutomaton):
        self.graph = graph
        self.regular_automaton = regular_automaton
        self.intersection_matrices = dict()
        self.__create_intersection_matrices__()

    def __create_intersection_matrices__(self):
        num_vert_graph = self.graph.get_number_of_vertices()
        num_vert_regex = self.regular_automaton.num_states
        num_verts_inter = num_vert_graph * num_vert_regex

        for symbol in self.regular_automaton.matrices:
            if symbol in self.graph:
                self.intersection_matrices[symbol] = Matrix.sparse(
                    BOOL, num_verts_inter, num_verts_inter
                )

    def __to_automaton__(self) -> EpsilonNFA:
        """
        Build automata from matrices
        """
        enfa = EpsilonNFA()
        graph_vertices_num = self.graph.get_number_of_vertices()

        start_states = [
            self.to_inter_coord(x, y)
            for x, y in product(
                range(graph_vertices_num), self.regular_automaton.start_states
            )
        ]

        final_states = [
            self.to_inter_coord(x, y)
            for x, y in product(
                range(graph_vertices_num), self.regular_automaton.final_states
            )
        ]

        for start_state in start_states:
            enfa.add_start_state(State(start_state))

        for final_state in final_states:
            enfa.add_final_state(State(final_state))

        for symbol in self.intersection_matrices:
            matrix = self.intersection_matrices[symbol]

            for row, col in zip(matrix.rows, matrix.cols):
                enfa.add_transition(State(row), Symbol(symbol), State(col))

        return enfa

    def to_inter_coord(self, graph_vert, reg_vert) -> int:
        """
        Converts coordinates of graph vertice and regex vertice
        to intersection coordinates vertice
        """
        return reg_vert * self.graph.get_number_of_vertices() + graph_vert

    def create_inverse_matrices(self) -> Dict[str, Matrix]:
        """
        Create inverse matrices from graph and regex matrices for each symbol
        """
        num_vert_regex = self.regular_automaton.num_states

        diag_matrices = dict()
        for symbol in self.regular_automaton.matrices:
            diag_matrices[symbol] = self.regular_automaton.matrices[symbol].dup().T

        return diag_matrices

    def create_masks_matrix(self) -> Matrix:
        num_vert_graph = self.graph.get_number_of_vertices()
        num_vert_regex = self.regular_automaton.num_states
        num_verts_diag = num_vert_graph + num_vert_regex

        mask_matrix = Matrix.sparse(BOOL, num_vert_regex, num_verts_diag)
        mask_matrix.assign_matrix(
            Matrix.identity(BOOL, num_vert_regex, value=True),
            slice(0, num_vert_regex - 1),
            slice(num_vert_graph, num_verts_diag - 1),
        )

        return mask_matrix
    def intersect_bfs(self, src_verts) -> EpsilonNFA:
        """
        Intersection implementation with synchronous breadth first traversal
        of a graph and regular grammar represented in automata
        """
        num_vert_graph = self.graph.get_number_of_vertices()
        num_vert_regex = self.regular_automaton.num_states

        num_verts_inter = num_vert_graph * num_vert_regex
        num_verts_diag = num_vert_graph + num_vert_regex

        graph = self.graph
        regex = self.regular_automaton.matrices
        mask_test =  Matrix.from_lists(self.regular_automaton.final_states, [0] * len(self.regular_automaton.final_states), [True] * len(self.regular_automaton.final_states), num_vert_regex, 1).T

        regex_start_states = self.regular_automaton.start_states

        inverse_matrices = self.create_inverse_matrices()

        result = Matrix.sparse(BOOL, 1, num_vert_graph)

        # initialize matrices for multiple source bfs
        ident = Matrix.sparse(BOOL, num_vert_regex, num_vert_graph)
        vect = ident.dup()
        found = ident.dup()

        # fill start states
        for reg_start_state in regex_start_states:
            for gr_start_state in src_verts:
                found[reg_start_state, gr_start_state] = True

        # matrix which contains newly found nodes on each iteration
        found_on_iter = found.dup()

        # Algo's body
        not_empty = True
        level = 0
        while not_empty and level < num_verts_inter:
            not_empty_for_at_least_one_symbol = False

            vect.assign_matrix(found_on_iter)

            # stores found nodes for each symbol

            for symbol in regex:
                with BOOL.ANY_PAIR:
                    vect.mxm(self.graph[symbol], out=found)
                    with Accum(binaryop.MAX_BOOL):
                        inverse_matrices[symbol].mxm(found, out=found_on_iter)
            if not found_on_iter.iseq(vect):
                not_empty_for_at_least_one_symbol = True

            with Accum(binaryop.MAX_BOOL):
                mask_test.mxm(found_on_iter, out=result)

            not_empty = not_empty_for_at_least_one_symbol
            level += 1

        return result

