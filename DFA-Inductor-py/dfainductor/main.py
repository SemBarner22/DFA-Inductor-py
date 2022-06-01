import queue
import sys
from collections import defaultdict
from multiprocessing.managers import DictProxy, BaseManager

import click

from . import examples
from .__about__ import __version__
from .algorithms.searchers import LSUS
from .logging_utils import *
from .statistics import STATISTICS
from .structures import APTA, InconsistencyGraph, DFA
import pebble as pb
import multiprocess as mp
from threading import Condition

p = None
amountCegars = 4
amountIter = 2
futures = set()
answerDfa = None
unsat = set()

class MyPriorityQueue(queue.PriorityQueue):
    def get_attribute(self, name):
        return getattr(self, name)

class MyManager(BaseManager):
    pass


MyManager.register('defaultdict', defaultdict, DictProxy)
MyManager.register("PriorityQueue", MyPriorityQueue)

def task_done(future):
    (dka, q_final) = future.result()
    print(str(dka))
    futures.remove(future)
    for a in futures:
        a.cancel()
    if dka is not None:
        print("found")
        global answerDfa
        answerDfa = dka
        q_final.put("answer")


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', 'input_', metavar='<PATH>', required=True, type=click.Path(exists=True),
              help='a DFA learning input file in Abbadingo format')
@click.option('-l', '--lower-bound', metavar='<INT>', type=int, default=1, show_default=True,
              help='lower bound of the DFA size')
# Поменял на время тестов с 100 до 18
@click.option('-u', '--upper-bound', metavar='<INT>', type=int, default=23, show_default=True,
              help='upper bound of the DFA size')
@click.option('-o', '--output', metavar='<PATH>', type=click.Path(allow_dash=True),
              help='write the found DFA using DOT language in <PATH> file; if not set, write to logging destination')
@click.option('-b', '--sym-breaking', type=click.Choice(['BFS', 'NOSB', 'TIGHTBFS']), default='BFS', show_default=True,
              help='symmetry breaking strategies')
# TODO: implement timeout
# @click.option('-t', '--timeout', metavar='<SECONDS>', type=int, help='set timeout')
@click.option('-s', '--solver', metavar='<SOLVER>', required=True, help='solver name')
@click.option('-cegar', '--cegar-mode', type=click.Choice(['none', 'lin-abs', 'lin-rel', 'geom']), default='none',
              show_default=True,
              help='counterexamples providing mode for CEGAR')
@click.option('-init', '--initial-amount', metavar='<INT>', type=int, default=50, show_default=True,
              help='initial amount of examples for CEGAR')
@click.option('-step', '--step-amount', metavar='<INT>', type=int, default=50, show_default=True,
              help='amount of examples added on each step for CEGAR')
@click.option('-pc', '--parallel-cegar', 'parallel_cegar', default=1, show_default=True,
              help='use several processes for CEGAR')
@click.option('-pd', '--parallel-dfa', 'parallel_dfa', default=1, show_default=True,
              help='use several processes for Iterating process')
@click.option('-a', '--assumptions', 'assumptions_mode', type=click.Choice(['none', 'switch', 'chain']),
              default='none', show_default=True, help='assumptions mode')
@click.option('-f', '--file', 'filename')
@click.option('-stat', '--statistics', 'print_statistics', is_flag=True, default=False, show_default=True,
              help='prints time statistics summary in the end')
@click.option('-ig', '--inconsistency-graph', 'use_ig', is_flag=True, default=False, show_default=True,
              help='use inconsistency graph')
@click.version_option(__version__, '-v', '--version')
def cli(input_: str,
        lower_bound: int,
        upper_bound: int,
        output: Optional[str],
        sym_breaking: str,
        solver: str,
        cegar_mode: str,
        parallel_cegar: int,
        parallel_dfa: int,
        filename: str,
        initial_amount: Optional[int],
        step_amount: Optional[int],
        assumptions_mode: str,
        print_statistics: bool,
        use_ig: bool) -> None:
    STATISTICS.start_whole_timer()
    global p
    global amount
    global index_futures
    amount = parallel_cegar * parallel_dfa
    p = pb.ProcessPool(amount)
    examples_ = []
    with click.open_file(input_) as file:
        examples_number, alphabet_size = [int(x) for x in next(file).split()]
        for __ in range(examples_number):
            examples_.append(next(file))
    manager = MyManager()
    manager2 = mp.Manager()
    manager.start()
    unsat = manager2.dict()
    state_amount = manager2.Value('state_amount', upper_bound)
    q_final = manager2.Queue()
    # q_final.put("a")
    # print(str(q_final.get_attribute("queue")))
    cond_for_processes = manager.PriorityQueue()
    global answerDfa
    if parallel_cegar > 1:
        current_amount = lower_bound
        solvers = ["Minicard", "Glucose4", "Minisat22", "MapleChrono"]
        cegars = ['none', 'none']
        if current_amount >= upper_bound:
            pass
        # if not use parallelDKA - current_amount,
        # current_amount + 1,
        else:
            for i in range(parallel_cegar):
                print(str(parallel_dfa))
                future = p.schedule(
                    start_procedure,
                    args=(
                        input_,
                        examples_,
                        lower_bound,
                        upper_bound,
                        output,
                        sym_breaking,
                        solvers[i],
                        cegars[i],
                        initial_amount,
                        step_amount,
                        assumptions_mode,
                        use_ig,
                        parallel_cegar,
                        parallel_dfa,
                        q_final,
                        unsat,
                        cond_for_processes,
                        state_amount
                    )
                )
                future.add_done_callback(task_done)
                futures.add(future)
        while True:
            a = q_final.get(block=True)
            if a == "answer":
                print("answer")
                break
            if a == "finishing":
                print("finishing")
                for i in range(parallel_cegar):
                    cond_for_processes.put("stopping")
            print("got in main " + str(a))
            if a == "ended":
                print("ended")
                for i in range(parallel_cegar):
                    cond_for_processes.put(str(i) + " to process")
        p.close()
        p.stop()
        p.join()
    else:
        answerDfa = start_procedure(
            input_,
            examples_,
            lower_bound,
            upper_bound,
            output,
            sym_breaking,
            solver,
            cegar_mode,
            initial_amount,
            step_amount,
            assumptions_mode,
            use_ig,
            parallel_cegar,
            parallel_dfa,
            q_final,
            unsat,
            cond_for_processes,
            state_amount
        )

    print("value is " + str(state_amount.value))
    print("len is " + str(len(unsat)))
    if answerDfa is None:
        log_error('DFA with less than {0} states not consistent with the given examples.'.format(upper_bound))
    else:
        if not output:
            log_br()
            log_info(str(answerDfa))
        else:
            log_info('Dumping found DFA to {0}'.format(output))
            try:
                with click.open_file(output, mode='w') as file:
                    file.write(str(answerDfa))
            except IOError as err:
                log_error('Something wrong with an output file: \'{0}\': {1}'.format(output, err))
                log_info('Dumping found DFA to console instead.')
                log_br()
                log_info(str(answerDfa))
        if answerDfa.check_consistency(examples_):
            log_success('DFA is consistent with the given examples.')
        else:
            log_error('DFA is not consistent with the given examples.')
    STATISTICS.stop_whole_timer()
    if print_statistics:
        STATISTICS.print_statistics(filename)

def start_procedure(
        input_: str,
        examples_: [],
        lower_bound: int,
        upper_bound: int,
        output: Optional[str],
        sym_breaking: str,
        solver: str,
        cegar_mode: str,
        initial_amount: Optional[int],
        step_amount: Optional[int],
        assumptions_mode: str,
        use_ig: bool,
        parallel_cegar: int,
        amountIter: int,
        q_final,
        unsat,
        cond_for_processes,
        state_amount
):
    examples_provider = examples.get_examples_provider(examples_, cegar_mode, initial_amount, step_amount)
    try:
        STATISTICS.start_apta_building_timer()
        apta = APTA(examples_provider.get_init_examples())
        log_success('Successfully built an APTA from file \'{0}\''.format(input_))
        log_info('The APTA size: {0}'.format(apta.size))
        STATISTICS.stop_apta_building_timer()
    except IOError as err:
        log_error('Cannot build an APTA from file \'{0}\': {1}'.format(input_, err))
        sys.exit(err.errno)
    if use_ig:
        STATISTICS.start_ig_building_timer()
    ig = InconsistencyGraph(apta, is_empty=not use_ig)
    if use_ig:
        log_success('Successfully built an IG')
        STATISTICS.stop_ig_building_timer()

    searcher = LSUS(apta,
                    ig,
                    solver,
                    sym_breaking,
                    cegar_mode,
                    examples_provider,
                    assumptions_mode)
    dfa = searcher.search(lower_bound, upper_bound, amountIter, q_final, unsat, cond_for_processes, state_amount)
    if not dfa:
        log_info('There is no such DFA.')
    else:
        if not output:
            log_br()
            log_info(str(dfa))
        else:
            log_info('Dumping found DFA to {0}'.format(output))
            try:
                with click.open_file(output, mode='w') as file:
                    file.write(str(dfa))
            except IOError as err:
                log_error('Something wrong with an output file: \'{0}\': {1}'.format(output, err))
                log_info('Dumping found DFA to console instead.')
                log_br()
                log_info(str(dfa))
        if dfa.check_consistency(examples_provider.get_all_examples()):
            log_success('DFA is consistent with the given examples.')
        else:
            log_error('DFA is not consistent with the given examples.')
    return dfa, q_final

