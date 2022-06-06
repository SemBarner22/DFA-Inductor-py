import queue
import re
import threading
from collections import defaultdict
from copy import copy
from typing import List

from dfainductor import main
from pysat.solvers import Solver, Glucose3
from pysat.parallel_solvers import ParallelSolverPathToFile, ParallelSolverPortfolio
from ..variables import VarPool
from .reductions import ClauseGenerator, _iff_conjunction_to_clauses
from ..examples import BaseExamplesProvider
from ..logging_utils import *

from ..statistics import STATISTICS
from ..structures import APTA, DFA, InconsistencyGraph
import pebble as pb
import time
import multiprocess as mp
import random


amount = 8
# amount = mp.cpu_count()
var: int = 1
index_futures = defaultdict(list)
size_dfas = defaultdict(list)
lock = threading.Lock()
# q_for_lock = queue.Queue()
res = False
dka = None
answer_found = False

data = threading.local()


class LSUS:
    _solver: Solver

    def __init__(self,
                 apta: APTA,
                 ig: InconsistencyGraph,
                 solver_name: str,
                 sb_strategy: str,
                 cegar_mode: str,
                 examples_provider: BaseExamplesProvider,
                 assumptions_mode: str,
                 used_colour: bool,
                 parallel_solver_path, parallel_solver_file) -> None:
        self._apta = apta
        self._ig = ig
        self._solver_name = solver_name
        self._sb_strategy = sb_strategy
        self._cegar_mode = cegar_mode
        self._examples_provider = examples_provider
        self._assumptions_mode = assumptions_mode
        manager = mp.Manager()
        self.cond_for_uc = manager.Queue()
        self.parallel_solver_path = parallel_solver_path
        self.parallel_solver_file = parallel_solver_file

        self._var_pool: VarPool = VarPool()
        self._used_colour = used_colour
        self._clause_generator = ClauseGenerator(self._apta,
                                                 self._ig,
                                                 self._var_pool,
                                                 self._assumptions_mode,
                                                 self._sb_strategy)

    def _try_to_synthesize_dfa(self, size: int, assumptions: List[int], variable, cond_for_uc):
        log_info('Vars in CNF: {0}'.format(self._solver.nof_vars()))
        log_info('Clauses in CNF: {0}'.format(self._solver.nof_clauses()))

        self._clause_generator.add_color_used_variables(8, self._solver, self._apta)

        STATISTICS.start_solving_timer()
        is_sat = self._solver.solve()
        print(str(self._solver.get_core()))
        print(is_sat)
        STATISTICS.stop_solving_timer()

        if is_sat:
            assignment = self._solver.get_model()

            dfa = DFA()
            for i in range(variable):
                dfa.add_state(
                    DFA.State.StateStatus.from_bool(assignment[self._var_pool.var('z', i) - 1] > 0)
                )
            for i in range(variable):
                for label in range(self._apta.alphabet_size):
                    for j in range(variable):
                        if assignment[self._var_pool.var('y', i, label, j) - 1] > 0:
                            dfa.add_transition(i, self._apta.alphabet[label], j)

            if self._used_colour:
                cond_for_uc.put(dfa)
            # if variable:
                # tuple(-self._var_pool.var('y', parent, l_less, child) for l_less in range(l_num))
                # self._solver.append_formula(
                # log_info(str(
                #         (-self._var_pool.var('x', jj, variable - 1) for jj in range(self._apta.size))
                # ))
                # )
                self._solver.add_clause(
                    (-self._var_pool.var('uc', variable),)
                )

                print(self._solver.nof_clauses())
                STATISTICS.start_solving_timer()
                is_sat1 = self._solver.solve()
                print(is_sat1)
                STATISTICS.stop_solving_timer()

                if is_sat1:
                    assignment = self._solver.get_model()
                    dfa = DFA()
                    variable -= 1
                    for i in range(variable):
                        dfa.add_state(
                            DFA.State.StateStatus.from_bool(assignment[self._var_pool.var('z', i) - 1] > 0)
                        )
                    for i in range(variable):
                        for label in range(self._apta.alphabet_size):
                            for j in range(variable):
                                if assignment[self._var_pool.var('y', i, label, j) - 1] > 0:
                                    dfa.add_transition(i, self._apta.alphabet[label], j)
                else:
                    print(str(self._solver.get_core()))
                    # return None

            # log_info("assignment" + str(assignment))


            return dfa, variable
        else:
            return None, variable

    def check_sat(self, variable):
        print('checking main.get_unsat()')
        flag = True
        for i in range(1, variable):
            if i not in self.unsat:
                print(str(i) + 'not in main.get_unsat()')
                flag = False
        return flag

    def task_done(self, future):
        global dka, index_futures, current_used
        put = False
        self.cond_for_uc.put(None)
        (dfa, variable) = future.result()
        if variable < 1:
            return

        try:
            lock.acquire(blocking=True)
            if future in index_futures[variable]:
                index_futures[variable].remove(future)
            if (len(index_futures[variable])) == 0:
                del index_futures[variable]
        finally:
            lock.release()

        if dfa is not None:
            try:
                lock.acquire(blocking=True)
                for a in index_futures.keys():
                    if a >= variable or a in self.unsat:
                        for f in index_futures[a]:
                            f.cancel()
                        # del index_futures[a]
                        # variable - 1
                # print(str("task_done(task, future)"))
                self.state_amount.value = min(self.state_amount.value, variable)
                if self.state_amount.value == variable:
                    dka = dfa
                if self.check_sat(variable):
                    put = True
                    self.q.put("finishing")
            finally:
                lock.release()
        else:
            self.unsat[variable] = 1

        print(str(self.state_amount.value))
        if self.check_sat(self.state_amount.value):
                #and dka is not None and dka.size() == self.state_amount.value
                #not put:
            put = True
            self.q.put("finishing")
        if not put:
            self.q.put("ended")

    def index_length(self):
        res = 0
        try:
            lock.acquire(blocking=True)
            for a in index_futures.keys():
                for aa in index_futures[a]:
                    if "cancel" in str(aa):
                        index_futures[a].remove(aa)
                    else:
                        res += 1
        finally:
            lock.release()
        print("result is " + str(res))
        return res

    def search(self, lower_bound: int, upper_bound: int, amountIter: int, q_final, unsat,
               cond_for_processes, state_amount) -> Optional[DFA]:
        global var, helper, dka, answer_found
        data.lower = lower_bound
        data.current_used = 0
        self.current_used = data.current_used
        data.pp = pb.ProcessPool(amountIter)
        self.q = q_final
        self.state_amount = state_amount
        self.unsat = unsat
        self.done = 1
        global index_futures
        print(str(data.pp))
        for i in range(amountIter):
            self.current_used += 1
            future = data.pp.schedule(
                helper.find_dka,
                args=(
                    data.lower,
                    lower_bound,
                    self
                )
            )
            future.add_done_callback(self.task_done)
            try:
                lock.acquire(blocking=True)
                index_futures.setdefault(data.lower, []).append(future)
            finally:
                lock.release()
            data.lower += 1
        while True:
            print(str(index_futures))
            print(str(cond_for_processes))
            t = self.cond_for_uc.get(block=True)
            if t is not None:
                global size_dfas
                print("daaamn")
                size_dfas[t.size()] = t
                # if t.size
            k = cond_for_processes.get(block=True)
            # if cond_for_processes.get_attribute("queue")[0] <= self.done:
            self.done += 1
            if k == "stopping":
                data.pp.close()
                data.pp.stop()
                data.pp.join()
                if self.check_sat(self.state_amount.value) and dka is not None and dka.size() == self.state_amount.value:
                    return dka
            print("got in process: " + str(k))
            try:
                lock.acquire(blocking=True)
                for a in index_futures.keys():
                    # if a >= self.state_amount.value and a in unsat:
                    if a in self.unsat or a >= self.state_amount.value:
                        for f in index_futures[a]:
                            f.cancel()
                        # del index_futures[a]
                        # variable - 1
                # print(str("task_done(task, future)"))
            finally:
                lock.release()
            if self.state_amount.value > data.lower:
                for i in range(amountIter - self.index_length()):
                    future = data.pp.schedule(
                        helper.find_dka,
                        args=(
                            data.lower,
                            lower_bound,
                            self
                        )
                    )
                    future.add_done_callback(self.task_done)
                    try:
                        lock.acquire(blocking=True)
                        index_futures.setdefault(data.lower, []).append(future)
                    finally:
                        lock.release()
                    data.lower += 1
                self.current_used += 1
            else:
                for i in range(amountIter - self.index_length()):
                    future = data.pp.schedule(
                        helper.find_dka,
                        args=(
                            self.state_amount.value - i - 1,
                            lower_bound,
                            self
                        )
                    )
                    future.add_done_callback(self.task_done)
                    try:
                        lock.acquire(blocking=True)
                        index_futures.setdefault(data.lower, []).append(future)
                    finally:
                        lock.release()
                    data.lower += 1
                self.current_used += 1
        first = True
        return dka


def correct_solver(solver_name, _parallel_solver_path, parallel_solver_file):
    if len(solver_name) > 0:
        if len(solver_name) == 1:
            #print("solver")
            _solver = Solver(solver_name[0])
        else:
            #print("portfolio")
            _solver = ParallelSolverPortfolio(solver_name)
    else:
        #print("path")
        _solver = ParallelSolverPathToFile(_parallel_solver_path[0], _parallel_solver_path[1],
                                           parallel_solver_file)
    return _solver


class Helper:

    def find_dka(self, variable: int, lower_bound: int,
                 value,
                 random_seed=None,
                 ):
        if variable < 1:
            return None, variable
        value._solver = correct_solver(value._solver_name, value.parallel_solver_path, value.parallel_solver_file)
        log_info('Solver has been started.')
        variable1 = variable
        for size in range(variable1, variable1 + 1):
            if value._assumptions_mode == 'none' and size > lower_bound:
                value._solver = correct_solver(value._solver_name, value.parallel_solver_path,
                                                     value.parallel_solver_file)
                log_info('Solver has been restarted.')
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))

            STATISTICS.start_formula_timer()
            if value._assumptions_mode != 'none' and size > lower_bound:
                log_info('_assumptions_mode isnt none')
                value._clause_generator.generate_with_new_size(value._solver, size - 1, size)
            else:
                log_info('_assumptions_mode is none')
                value._clause_generator.generate(value._solver, size)
            STATISTICS.stop_formula_timer()
            assumptions = value._clause_generator.build_assumptions(size, value._solver)
            while True:
                dfa, variable = value._try_to_synthesize_dfa(size, assumptions, variable, value.cond_for_uc)
                if dfa:
                    if not value.provider.is_provider_calcs_cover():
                        counter_examples = value.provider.get_counter_examples(dfa)
                    else:
                        counter_examples = value.provider.get_cover(dfa)
                    if len(counter_examples) > 0:
                        print("cover" + str(counter_examples[-1]))
                    if counter_examples:
                        log_info('An inconsistent DFA with {0} states is found.'.format(variable))
                        log_info('Added {0} counterexamples.'.format(len(counter_examples)))

                        STATISTICS.start_apta_building_timer()
                        (new_nodes_from, changed_statuses) = value._apta.add_examples(counter_examples)
                        STATISTICS.stop_apta_building_timer()

                        STATISTICS.start_ig_building_timer()
                        value._ig.update(new_nodes_from)
                        STATISTICS.stop_ig_building_timer()

                        STATISTICS.start_formula_timer()
                        value._clause_generator.generate_with_new_counterexamples(value._solver, variable,
                                                                                 new_nodes_from,
                                                                                 changed_statuses)
                        STATISTICS.stop_formula_timer()
                        continue
                break
            if not dfa:
                log_info('Not found a DFA with {0} states.'.format(size))
            else:
                log_success('The DFA with {0} states is found!'.format(size))
            print("Done")
            # if not check_sat(variable):
            #     _clause_generator.add_color_used_variables(variable, value._solver, value._apta)
            return dfa, variable


helper = Helper()
