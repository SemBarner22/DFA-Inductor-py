import queue
from typing import List

from pysat.solvers import Solver, Glucose3

from ..variables import VarPool
from .reductions import ClauseGenerator
from ..examples import BaseExamplesProvider
from ..logging_utils import *
from ..statistics import STATISTICS
from ..structures import APTA, DFA, InconsistencyGraph
import pebble as pb
import time
import multiprocess as mp
import random


amount = 8
window = 3
# amount = mp.cpu_count()
p = None
var: int = 1
index_futures = dict()

lower = 0
q_final = queue.Queue()
q_for_lock = queue.Queue()
res = False
dka = None
state_amount = 1000
answer_found = False
unsat = set()


def quit(i):
    print("quit")
    if i is None:
        global p, var
        print(str(var))
        var += 1
        if var < 100:
            pass
        else:
            q_for_lock.put("d")
            p.terminate()
        print("end")
        return None
    return None


class LSUS:
    _solver: Solver

    def __init__(self,
                 apta: APTA,
                 ig: InconsistencyGraph,
                 solver_name: str,
                 sb_strategy: str,
                 cegar_mode: str,
                 examples_provider: BaseExamplesProvider,
                 assumptions_mode: str) -> None:
        self._apta = apta
        self._ig = ig
        self._solver_name = solver_name
        self._sb_strategy = sb_strategy
        self._cegar_mode = cegar_mode
        self._examples_provider = examples_provider
        self._assumptions_mode = assumptions_mode

        self._var_pool: VarPool = VarPool()
        self._clause_generator = ClauseGenerator(self._apta,
                                                 self._ig,
                                                 self._var_pool,
                                                 self._assumptions_mode,
                                                 self._sb_strategy)

    def _try_to_synthesize_dfa(self, size: int, assumptions: List[int]) -> Optional[DFA]:
        log_info('Vars in CNF: {0}'.format(self._solver.nof_vars()))
        log_info('Clauses in CNF: {0}'.format(self._solver.nof_clauses()))

        STATISTICS.start_solving_timer()
        is_sat = self._solver.solve()
        print(is_sat)
        STATISTICS.stop_solving_timer()

        if is_sat:
            assignment = self._solver.get_model()
            log_info("assignment" + str(assignment))
            dfa = DFA()
            for i in range(size):
                dfa.add_state(
                    DFA.State.StateStatus.from_bool(assignment[self._var_pool.var('z', i) - 1] > 0)
                )
            for i in range(size):
                for label in range(self._apta.alphabet_size):
                    for j in range(size):
                        if assignment[self._var_pool.var('y', i, label, j) - 1] > 0:
                            dfa.add_transition(i, self._apta.alphabet[label], j)
            return dfa
        else:
            return None

    def search(self, lower_bound: int, upper_bound: int) -> Optional[DFA]:
        global var, p, helper, index_futures, dka, lower, answer_found
        lower = lower_bound
        p = pb.ProcessPool(amount)
        for i in range(window):
            future = p.schedule(
                helper.find_dka,
                args=(
                    var,
                    lower_bound,
                    self._solver_name,
                    self._assumptions_mode,
                    self._clause_generator,
                    self
                )
            )
            future.add_done_callback(task_done)
            index_futures.setdefault(var, []).append(future)
            var += 1
        first = True
        global state_amount
        while not answer_found:
            ans = q_for_lock.get(block=True)
            if ans == "yes" and not answer_found and first and state_amount != 1000:
                first = False
                print("found large")
                v = state_amount - window
                names = ["Glucose4", "Minicard", "Lingeling", "Minisat22"]
                for i in range(amount // 2):
                    future = p.schedule(
                        helper.find_dka,
                        args=(
                            v,
                            lower,
                            names[i],
                            self._assumptions_mode,
                            self._clause_generator,
                            self
                        )
                    )
                    future.add_done_callback(task_done)
                    index_futures.setdefault(v, []).append(future)
                v = v + 1
                for i in range(amount // 2):
                    future = p.schedule(
                        helper.find_dka,
                        args=(
                            v,
                            lower,
                            names[i],
                            self._assumptions_mode,
                            self._clause_generator,
                            self
                        )
                    )
                    future.add_done_callback(task_done)
                    index_futures.setdefault(v, []).append(future)
            else:
                print("not found")
                if var < state_amount:
                    future = p.schedule(
                        helper.find_dka,
                        args=(
                            var,
                            lower,
                            self._solver_name,
                            self._assumptions_mode,
                            self._clause_generator,
                            self
                        )
                    )
                    future.add_done_callback(task_done)
                    index_futures.setdefault(var, []).append(future)
                    var = var + 1
        p.close()
        p.stop()
        p.join()
        return dka

class Helper:

    def find_dka(self, variable: int, lower_bound: int, _solver_name,
                 _assumptions_mode,
                 _clause_generator,
                 value,
                 random_seed=None
                 ) -> Optional[DFA]:
        print("find dkaaaa" + str(variable))
        # if variable > 100:
        #     return None
        value._solver = Solver(_solver_name)
        # random.randint(1, 250)
        # value._solver = Glucose3(random_seed=236)
        log_info('Solver has been started.')
        for size in range(variable, variable + 1):
            if _assumptions_mode == 'none' and size > lower_bound:
                # value._solver = Glucose3(random_seed=643)
                value._solver = Solver(_solver_name)
                # value._solver = Glucose3(random_seed=random.randint(1, 250))
                log_info('Solver has been restarted.')
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))

            STATISTICS.start_formula_timer()
            if _assumptions_mode != 'none' and size > lower_bound:
                log_info('_assumptions_mode isnt none')
                _clause_generator.generate_with_new_size(value._solver, size - 1, size)
            else:
                log_info('_assumptions_mode is none')
                _clause_generator.generate(value._solver, size)
            STATISTICS.stop_formula_timer()
            assumptions = _clause_generator.build_assumptions(size, value._solver)
            while True:
                dfa = value._try_to_synthesize_dfa(size, assumptions)
                if dfa:
                    counter_examples = value._examples_provider.get_counter_examples(dfa)
                    if counter_examples:
                        log_info('An inconsistent DFA with {0} states is found.'.format(size))
                        log_info('Added {0} counterexamples.'.format(len(counter_examples)))

                        STATISTICS.start_apta_building_timer()
                        (new_nodes_from, changed_statuses) = value._apta.add_examples(counter_examples)
                        STATISTICS.stop_apta_building_timer()

                        STATISTICS.start_ig_building_timer()
                        value._ig.update(new_nodes_from)
                        STATISTICS.stop_ig_building_timer()

                        STATISTICS.start_formula_timer()
                        _clause_generator.generate_with_new_counterexamples(value._solver, size,
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
            #     print(str(value._solver.solve()) + "ergfergregergergerg")
            return dfa, variable

helper = Helper()


def check_sat(variable):
    global unsat
    print('checking unsat')
    flag = True
    for i in range(1, variable):
        if i not in unsat:
            print(str(i) + 'not in unsat')
            flag = False
    return flag


def task_done(future):
    global unsat, state_amount, dka, answer_found
    (dfa, variable) = future.result()
    print(str(dfa))
    print(str(variable))
    if dfa is not None:
        for a in index_futures.keys():
            if a >= variable:
                for f in index_futures[a]:
                    f.cancel()
                # variable - 1
        print(str("task_done(task, future)"))
        if state_amount > variable:
            dka = dfa
            state_amount = variable
        time.sleep(1)
        for a in index_futures.keys():
            for f in index_futures[a]:
                f.cancel()
        if check_sat(variable):
            answer_found = True
            q_final.put("k")
            dka = dfa
            for f in index_futures:
                for ff in index_futures[f]:
                    ff.cancel()
        q_for_lock.put("yes")
    else:
        if state_amount != 1000 and check_sat(state_amount):
            answer_found = True
            q_final.put("k")
            for f in index_futures:
                for ff in index_futures[f]:
                    ff.cancel()
        unsat.add(variable)
        q_for_lock.put("no")
