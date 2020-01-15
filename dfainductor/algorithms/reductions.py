from abc import ABC, abstractmethod

from pysat.card import CardEnc
from pysat.formula import CNF

from ..structures import APTA


def _implication_to_clauses(lhs, rhs):
    """
    generates CNF formula of an expression /lhs => rhs/
    :type lhs: int
    :type rhs: int
    """
    return [[-lhs, rhs]]


def _conjunction_implies_to_clauses(lhs, rhs):
    """
    generates CNF formula of an expression /lhs_1 and lhs_2 and ... and lhs_n => rhs/
    :type lhs: list(int)
    :type rhs: int
    """
    return [[-arg for arg in lhs] + [rhs]]


def _iff_to_clauses(lhs, rhs):
    """
    generates CNF formula of an expression /lhs <=> rhs/
    :type lhs: int
    :type rhs: int
    """
    return [[-lhs, rhs], [lhs, -rhs]]


def _iff_disjunction_to_clauses(lhs, rhs):
    """
    generates CNF formula of an expression /lhs <=> rhs_1 or rhs_2 or ... or rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    return [[-lhs] + (list(rhs))] + [[lhs, -arg] for arg in rhs]


def _iff_conjunction_to_clauses(lhs, rhs):
    """
    generates CNF formula of an expression /lhs <=> rhs_1 and rhs_2 and ... and rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    return [[lhs] + [-arg for arg in rhs]] + [[-lhs, arg] for arg in rhs]


class BaseClauseGenerator(ABC):
    def __init__(self, apta, dfa_size, vpool):
        self._apta = apta
        self._dfa_size = dfa_size
        self._vpool = vpool
        self._formula = CNF()
        self._alphabet = self._apta.alphabet
        self._alphabet_size = len(self._alphabet)

    @abstractmethod
    def generate(self):
        pass

    def _update_vpool_top(self, formula):
        if formula.nv > 0 and formula.nv > self._vpool.top:
            self._vpool.top = formula.nv

    def _var(self, name, *indices):
        var = name + '_' + '_'.join(str(index) for index in indices)
        return self._vpool.id(var)


class MinDFAToSATClausesGenerator(BaseClauseGenerator):
    def generate(self):
        formula = self._fix_start_state()
        # print(formula)
        self._formula.extend(formula)

        formula = self._one_node_maps_to_at_least_one_state()
        # print(formula)
        self._formula.extend(formula)

        formula = self._one_node_maps_to_at_most_one_state()
        # print(formula)
        self._formula.extend(formula)

        formula = self._dfa_is_complete()
        # print(formula)
        self._formula.extend(formula)

        formula = self._dfa_is_deterministic()
        # print(formula)
        self._formula.extend(formula)

        formula = self._state_status_compatible_with_node_status()
        # print(formula)
        self._formula.extend(formula)

        formula = self._mapped_adjacent_nodes_force_transition()
        # print(formula)
        self._formula.extend(formula)

        formula = self._mapped_node_and_transition_force_mapping()
        # print(formula)
        self._formula.extend(formula)

        return self._formula

    def _fix_start_state(self):
        clauses = [[self._var('x', 0, 0)]]
        return CNF(from_clauses=clauses)

    def _one_node_maps_to_at_least_one_state(self):
        formula = CNF()
        for i in range(self._apta.size()):
            formula.extend(
                CardEnc.atleast(
                    [self._var('x', i, j) for j in range(self._dfa_size)],
                    top_id=self._vpool.top
                )
            )
            self._update_vpool_top(formula)
        return formula

    def _one_node_maps_to_at_most_one_state(self):
        formula = CNF()
        for node in self._apta.nodes:
            formula.extend(
                CardEnc.atmost(
                    [self._var('x', node.id_, j) for j in range(self._dfa_size)],
                    top_id=self._vpool.top
                )
            )
            self._update_vpool_top(formula)
        return formula

    def _dfa_is_complete(self):
        formula = CNF()
        for i in range(self._dfa_size):
            for l_id in range(self._alphabet_size):
                formula.extend(
                    CardEnc.atleast(
                        [self._var('y', i, l_id, j) for j in range(self._dfa_size)],
                        top_id=self._vpool.top
                    )
                )
                self._update_vpool_top(formula)
        return formula

    def _dfa_is_deterministic(self):
        formula = CNF()
        for i in range(self._dfa_size):
            for l_id in range(self._alphabet_size):
                formula.extend(
                    CardEnc.atmost(
                        [self._var('y', i, l_id, j) for j in range(self._dfa_size)],
                        top_id=self._vpool.top
                    )
                )
                self._update_vpool_top(formula)
        return formula

    def _state_status_compatible_with_node_status(self):
        formula = CNF()
        for node in self._apta.nodes:
            if node.status is APTA.Node.NodeStatus.ACCEPTING:
                for j in range(self._dfa_size):
                    formula.extend(
                        _implication_to_clauses(
                            self._var('x', node.id_, j),
                            self._var('z', j)
                        )
                    )
            elif node.status is APTA.Node.NodeStatus.REJECTING:
                for j in range(self._dfa_size):
                    formula.extend(
                        _implication_to_clauses(
                            self._var('x', node.id_, j),
                            -self._var('z', j)
                        )
                    )
        return formula

    def _mapped_adjacent_nodes_force_transition(self):
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if child:
                    for from_ in range(self._dfa_size):
                        for to in range(self._dfa_size):
                            formula.extend(
                                _conjunction_implies_to_clauses(
                                    [
                                        self._var('x', parent.id_, from_),
                                        self._var('x', child.id_, to),
                                    ],
                                    self._var('y', from_, label, to)
                                )
                            )
        return formula

    def _mapped_node_and_transition_force_mapping(self):
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if child:
                    for from_ in range(self._dfa_size):
                        for to in range(self._dfa_size):
                            formula.extend(
                                _conjunction_implies_to_clauses(
                                    [
                                        self._var('x', parent.id_, from_),
                                        self._var('y', from_, label, to),
                                    ],
                                    self._var('x', child.id_, to)
                                )
                            )
        return formula


class BFSBasedSymBreakingClausesGenerator(BaseClauseGenerator):
    def generate(self):
        formula = self._define_t_variables()
        self._formula.extend(formula)

        formula = self._define_p_variables()
        self._formula.extend(formula)

        formula = self._state_has_at_least_one_parent()
        self._formula.extend(formula)

        formula = self._preserve_parent_order_on_children()
        self._formula.extend(formula)

        formula = self._order_children()
        self._formula.extend(formula)
        return self._formula

    def _define_t_variables(self):
        formula = CNF()
        for from_ in range(self._dfa_size):
            for to in range(from_ + 1, self._dfa_size):
                formula.extend(
                    _iff_disjunction_to_clauses(
                        self._var('t', from_, to),
                        [self._var('y', from_, l_id, to) for l_id in range(self._alphabet_size)]
                    )
                )
        return formula

    def _define_p_variables(self):
        formula = CNF()
        for parent in range(self._dfa_size):
            for child in range(parent + 1, self._dfa_size):
                formula.extend(
                    _iff_conjunction_to_clauses(
                        self._var('p', child, parent),
                        [-self._var('t', prev, child) for prev in range(parent)] + [self._var('t', parent, child)]
                    )
                )
        return formula

    def _state_has_at_least_one_parent(self):
        formula = CNF()
        for child in range(1, self._dfa_size):
            formula.extend(
                CardEnc.atleast(
                    [self._var('p', child, parent) for parent in range(child)],
                    top_id=self._vpool.top
                )
            )
            self._update_vpool_top(formula)
        return formula

    def _preserve_parent_order_on_children(self):
        formula = CNF()
        for child in range(2, self._dfa_size - 1):
            for parent in range(1, child):
                for pre_parent in range(parent):
                    formula.extend(
                        _implication_to_clauses(self._var('p', child, parent), -self._var('p', child + 1, pre_parent))
                    )
        return formula

    def _order_children(self):
        formula = CNF()
        if self._alphabet_size == 2:
            formula.extend(self._order_children_with_binary_alphabet())
        elif self._alphabet_size > 2:
            formula.extend(self._define_m_variables())
            formula.extend(self._order_children_using_m())
        return formula

    def _order_children_with_binary_alphabet(self):
        formula = CNF()
        for child in range(self._dfa_size - 1):
            for parent in range(child):
                formula.extend(
                    _conjunction_implies_to_clauses(
                        [self._var('p', child, parent), self._var('p', child + 1, parent)],
                        self._var('y', parent, self._alphabet[0], child)
                    )
                )
                formula.extend(
                    _conjunction_implies_to_clauses(
                        [self._var('p', child, parent), self._var('p', child + 1, parent)],
                        self._var('y', parent, self._alphabet[1], child + 1)
                    )
                )
        return formula

    def _define_m_variables(self):
        formula = CNF()
        for child in range(self._dfa_size):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    formula.extend(
                        _iff_conjunction_to_clauses(
                            self._var('m', parent, l_num, child),
                            [
                                -self._var('y', parent, l_less, child) for l_less in range(l_num)
                            ] + [self._var('y', parent, l_num, child)]
                        )
                    )
        return formula

    def _order_children_using_m(self):
        formula = CNF()
        for child in range(self._dfa_size - 1):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    for l_less in range(l_num):
                        formula.extend(
                            _conjunction_implies_to_clauses(
                                [
                                    self._var('p', child, parent),
                                    self._var('p', child + 1, parent),
                                    self._var('m', parent, l_num, child),
                                ],
                                -self._var('m', parent, l_less, child + 1)
                            )
                        )
        return formula
