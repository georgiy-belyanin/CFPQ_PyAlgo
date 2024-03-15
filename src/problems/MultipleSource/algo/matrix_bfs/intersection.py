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
        self.graph.load_bool_graph()
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

    def create_diag_matrices(self) -> Dict[str, Matrix]:
        """
        Create a block diagonal matrices from graph and regex matrices for each symbol
        """
        num_vert_graph = self.graph.get_number_of_vertices()
        num_vert_regex = self.regular_automaton.num_states
        diag_num_verts = num_vert_graph + num_vert_regex

        diag_matrices = dict()
        for symbol in self.regular_automaton.matrices:
            if symbol in self.graph:
                diag_matrix = Matrix.sparse(BOOL, diag_num_verts, diag_num_verts)
                diag_matrix.assign_matrix(
                    self.regular_automaton.matrices[symbol],
                    slice(0, num_vert_regex - 1),
                    slice(0, num_vert_regex - 1),
                )
                diag_matrix.assign_matrix(
                    self.graph[symbol],
                    slice(num_vert_regex, diag_num_verts - 1),
                    slice(num_vert_regex, diag_num_verts - 1),
                )

                diag_matrices[symbol] = diag_matrix

        return diag_matrices
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

        mask_matrix = Matrix.identity(BOOL, num_vert_regex, value=True)
        mask_matrix.resize(num_vert_regex, num_verts_diag)

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

        regex_start_states = self.regular_automaton.start_states

        diag_matrices = self.create_diag_matrices()
        inverse_matrices = self.create_inverse_matrices()

        result = Matrix.sparse(BOOL, 1, num_vert_graph)

        src_verts_matrix = Matrix.from_lists(src_verts, [0] * len(src_verts), [True] * len(src_verts), num_vert_graph, 1)

        # initialize matrices for multiple source bfs
        ident = self.create_masks_matrix()
        vect = ident.dup()
        found = ident.dup()

        # fill start states
        for reg_start_state in regex_start_states:
            for gr_start_state in src_verts:
                found[reg_start_state, num_vert_regex + gr_start_state] = True

        # matrix which contains newly found nodes on each iteration
        found_on_iter = found.dup()

        # Algo's body
        not_empty = True
        level = 0
        while not_empty and level < num_verts_inter:
            # for each symbol we are going to store if any new nodes were found during traversal.
            # if none are found, then 'not_empty' flag turns False, which means that no matrices change anymore
            # and we can stop the traversal
            not_empty_for_at_least_one_symbol = False

            vect.assign_matrix(found_on_iter, mask=vect, desc=descriptor.RC)
            vect.assign_scalar(True, mask=ident)

            # stores found nodes for each symbol
            found_on_iter.assign_matrix(ident)

            for symbol in regex:
                if symbol in graph:
                    with BOOL.ANY_PAIR:
                        found = vect.mxm(diag_matrices[symbol])

                    with BOOL.ANY_PAIR:
                        found_on_iter += inverse_matrices[symbol].dup().mxm(found)
                    with Accum(binaryop.MAX_BOOL):
                        identity = Matrix.identity(BOOL, num_vert_regex)
                        identity.resize(num_vert_regex, num_vert_regex + num_vert_graph)
                        found_on_iter += identity
                    if not found_on_iter.iseq(vect):
                        not_empty_for_at_least_one_symbol = True

            with Accum(binaryop.MAX_BOOL):
                Matrix.dense(BOOL, 1, num_vert_regex, fill=1).mxm(found_on_iter.extract_matrix(col_index=slice(num_vert_regex, num_verts_diag - 1)), out=result, mask=src_verts_matrix.T, desc=descriptor.C) 

            not_empty = not_empty_for_at_least_one_symbol
            level += 1

        return result
