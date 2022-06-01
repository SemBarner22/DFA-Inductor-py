import queue
import re
import threading
from collections import defaultdict
from copy import copy
from typing import List

from dfainductor import main
from pysat.solvers import Solver, Glucose3

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

lock = threading.Lock()
# q_for_lock = queue.Queue()
res = False
dka = None
answer_found = False

data = threading.local()


# def quit(i):
#     print("quit")
#     if i is None:
#         global var
#         print(str(var))
#         var += 1
#         if var < 100:
#             pass
#         else:
#             q_for_lock.put("d")
#             p.terminate()
#         print("end")
#         return None
#     return None


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

    def _try_to_synthesize_dfa(self, size: int, assumptions: List[int], variable):
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

            # if variable == 13:
            #     log_info("8kekek8")
            #     # tuple(-self._var_pool.var('y', parent, l_less, child) for l_less in range(l_num))
            #     # self._solver.append_formula(
            #     # log_info(str(
            #     #         (-self._var_pool.var('x', jj, 7) for jj in range(self._apta.size))
            #     # ))
            #     # )
            #     self._solver.add_clause(
            #         (-self._var_pool.var('uc', 12),)
            #     )
            #
            #     print(self._solver.nof_clauses())
            #     STATISTICS.start_solving_timer()
            #     is_sat1 = self._solver.solve()
            #     print(is_sat1)
            #     print("UC DID SOMETHING YEAHHHH")
            #     STATISTICS.stop_solving_timer()
            #
            #     if is_sat1:
            #         print("UC DID FOUND OMG")
            #         assignment = self._solver.get_model()
            #         variable = 12
            #     else:
            #         print("UC DID NOT FOUND SAD")
            #         print(str(self._solver.get_core()))
            #         return None

            # log_info("assignment" + str(assignment))

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
        print("task done called lol")
        global dka, index_futures, current_used
        put = False
        # print("kekekkeekkekekekekeke")
        (dfa, variable) = future.result()
        if variable < 1:
            return

        try:
            lock.acquire(blocking=True)
            # print("size is " + str(len(index_futures[variable])))

            if future in index_futures[variable]:
                index_futures[variable].remove(future)
            if (len(index_futures[variable])) == 0:
                del index_futures[variable]

            # print("size is " + str(len(index_futures[variable])))

            # print("a " + str(index_futures))
        finally:
            lock.release()
        # print("len is:" + str(len(index_futures.keys())))

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

        print("aboooooooooooooooobaaaaa" + str(self.state_amount.value))
        if self.check_sat(self.state_amount.value):
                #and dka is not None and dka.size() == self.state_amount.value
                #not put:
            put = True
            self.q.put("finishing")
        if not put:
            self.q.put("ended")
        #     if state_amount > variable:
        #         dka = dfa
        #         state_amount = variable
        #     time.sleep(1)
        #     for a in index_futures.keys():
        #         for f in index_futures[a]:
        #             f.cancel()
        #     if check_sat(variable):
        #         answer_found = True
        #         main.get_q_final().put("k")
        #         dka = dfa
        #         for f in index_futures:
        #             for ff in index_futures[f]:
        #                 ff.cancel()
        #     q_for_lock.put("yes")
        # else:
        #     if state_amount != 1000 and check_sat(state_amount):
        #         answer_found = True
        #         main.get_q_final().put("k")
        #         for f in index_futures:
        #             for ff in index_futures[f]:
        #                 ff.cancel()
        #         finish(state_amount, dka)
        #     main.get_unsat().add(variable)
        #     q_for_lock.put("no")

    def index_length(self):
        res = 0
        try:
            lock.acquire(blocking=True)
            for a in index_futures.keys():
                for aa in index_futures[a]:
                    if "cancel" in str(aa):
                        print("EASIKOFUhwufhweliofuhwef")
                        index_futures[a].remove(aa)
                    else:
                        res += 1
        finally:
            lock.release()
        print("result is " + str(res))
        return res

    # def get_clauses(self, str):
    #     with open(str) as fifo:
    #         while True:
    #             for line in fifo:
    #                 #line
    #                 data = fifo.read()
    #                 #do_work(data)
    #
    #     return str

    def search(self, lower_bound: int, upper_bound: int, amountIter: int, q_final, unsat, cond_for_processes, state_amount) -> Optional[DFA]:
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
            print("hehe")
            self.current_used += 1
            future = data.pp.schedule(
                helper.find_dka,
                args=(
                    data.lower,
                    lower_bound,
                    self._solver_name,
                    self._assumptions_mode,
                    self._clause_generator,
                    self._examples_provider,
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
        # q_final.put("wefewff")
        while True:
            print(str(index_futures))
            print(str(cond_for_processes))
            k = cond_for_processes.get(block=True)
            # if cond_for_processes.get_attribute("queue")[0] <= self.done:
            self.done += 1
            if k == "stopping":
                print("kek")
                data.pp.close()
                data.pp.stop()
                data.pp.join()
                if self.check_sat(self.state_amount.value) and dka is not None and dka.size() == self.state_amount.value:
                    print("kok")
                    return dka
            print("got in process: " + str(k))
            # стопать те которые уже не нужны
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
                            self._solver_name,
                            self._assumptions_mode,
                            self._clause_generator,
                            self._examples_provider,
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
                            self._solver_name,
                            self._assumptions_mode,
                            self._clause_generator,
                            self._examples_provider,
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
        # while not answer_found:
        #     ans = q_for_lock.get(block=True)
        #     if ans == "yes" and not answer_found and first and state_amount != 1000:
        #         first = False
        #         print("found large")
        #         v = state_amount - window
        #         names = ["Glucose4", "Minicard", "Lingeling", "Minisat22"]
        #         for i in range(amount // 2):
        #             future = data.pp.schedule(
        #                 helper.find_dka,
        #                 args=(
        #                     v,
        #                     data.lower,
        #                     names[i],
        #                     self._assumptions_mode,
        #                     self._clause_generator,
        #                     self._examples_provider,
        #                     self
        #                 )
        #             )00
        #             future.add_done_callback(self.task_done)
        #             index_futures.setdefault(v, []).append(future)
        #         v = v + 1
        #         for i in range(amount // 2):
        #             future = data.pp.schedule(
        #                 helper.find_dka,
        #                 args=(
        #                     v,
        #                     data.lower,
        #                     names[i],
        #                     self._assumptions_mode,
        #                     self._clause_generator,
        #                     self._examples_provider,
        #                     self
        #                 )
        #             )
        #             future.add_done_callback(self.task_done)
        #             index_futures.setdefault(v, []).append(future)
        #     else:
        #         print("not found")
        #         if var < state_amount:
        #             future = data.pp.schedule(
        #                 helper.find_dka,
        #                 args=(
        #                     var,
        #                     data.lower,
        #                     self._solver_name,
        #                     self._assumptions_mode,
        #                     self._clause_generator,
        #                     self._examples_provider,
        #                     self
        #                 )
        #             )
        #             future.add_done_callback(self.task_done)
        #             index_futures.setdefault(var, []).append(future)
        #             var = var + 1

class Helper:

    def find_dka(self, variable: int, lower_bound: int, _solver_name,
                 _assumptions_mode,
                 _clause_generator,
                 provider,
                 value,
                 random_seed=None
                 ):
        if variable < 1:
            return None, variable
        # print(str(var) + "wefewfwf")
        # print("find dkaaaa" + str(variable))
        # if variable > 100:
        #     return None
        value._solver = Solver(_solver_name)
        # random.randint(1, 250)
        # value._solver = Glucose3(random_seed=236)
        log_info('Solver has been started.')
        variable1 = variable
        for size in range(variable1, variable1 + 1):
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
            check = 0
            while True:
                print("syntez" + str(check))
                check += 1
                dfa, variable = value._try_to_synthesize_dfa(size, assumptions, variable)
                if dfa:
                    #dfa._perform_cover_calculating(provider.examples)
                    counter_examples = provider.get_counter_examples(dfa)
                    #all_counter_examples = provider.get_all_counter_examples(dfa)
                    #counter_amount = min(len(provider.get_counter_examples(dfa)), len(all_counter_examples))
                    #counter_examples = sorted(all_counter_examples, key=lambda x: dfa.cover_for_word_count(x))[
                    #                    :counter_amount]
                    # variable = dfa.size()
                    if len(counter_examples) > 0:
                        print("cover" + str(counter_examples[-1]))
                    else:
                        print("zero ce")
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
                        _clause_generator.generate_with_new_counterexamples(value._solver, variable,
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

#
# def finish(dfa, q_final):
#     print("finishing")
#     main.set_answer(dfa)
#     q_final.put("finish")