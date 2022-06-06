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
@click.option('-u', '--upper-bound', metavar='<INT>', type=int, default=23, show_default=True,
              help='upper bound of the DFA size')
@click.option('-o', '--output', metavar='<PATH>', type=click.Path(allow_dash=True),
              help='write the found DFA using DOT language in <PATH> file; if not set, write to logging destination')
@click.option('-b', '--sym-breaking', type=click.Choice(['BFS', 'NOSB', 'TIGHTBFS']), default='BFS', show_default=True,
              help='symmetry breaking strategies')
# TODO: implement timeout
# @click.option('-t', '--timeout', metavar='<SECONDS>', type=int, help='set timeout')
@click.option('-s', '--solver', metavar='<SOLVER>', help='solver name', multiple=True, type=str, default=None)
@click.option('-ps', '--parallel-solver-path', 'parallel_solver_path', nargs=2, type=str,
              help='path to execute file, working directory', default=(None, None))
@click.option('-psf', '--parallel-solver-file', 'parallel_solver_file', is_flag=True, default=False,
              help='if parallel solver needs file to communicate with')
@click.option('-cegar', '--cegar-mode', multiple=True, type=str, default=list('none'),
              help='counterexamples providing mode for CEGAR. Should be either one of: '
                                   'none, lin-abs, lin-rel, geom,'
                                   'cover-lin-abs, cover-lin-rel, cover-geom')
@click.option('-init', '--initial-amount', metavar='<INT>', type=int, default=50, show_default=True,
              help='initial amount of examples for CEGAR')
@click.option('-step', '--step-amount', metavar='<INT>', type=int, default=50, show_default=True,
              help='amount of examples added on each step for CEGAR')
@click.option('-uc', '--used-colour', 'used_colour', is_flag=True, default=False, show_default=True,
              help='using colour variables')
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
        solver: Optional[str],
        parallel_solver_path,
        parallel_solver_file: bool,
        cegar_mode: Optional[str],
        parallel_dfa: int,
        filename: str,
        initial_amount: Optional[int],
        step_amount: Optional[int],
        assumptions_mode: str,
        print_statistics: bool,
        use_ig: bool,
        used_colour: bool) -> None:
    STATISTICS.start_whole_timer()
    global p
    global amount
    global index_futures
    amount = len(cegar_mode) * parallel_dfa
    p = pb.ProcessPool(amount)
    examples_ = []
    with click.open_file(input_) as file:
        examples_number, alphabet_size = [int(x) for x in next(file).split()]
        for __ in range(examples_number):
            examples_.append(next(file))
    manager = MyManager()
    parallel_cegar = len(cegar_mode)
    manager2 = mp.Manager()
    manager.start()
    unsat = manager2.dict()
    state_amount = manager2.Value('state_amount', upper_bound)
    q_final = manager2.Queue()
    cond_for_processes = manager.PriorityQueue()
    global answerDfa
    if len(cegar_mode) > 1:
        current_amount = lower_bound
        if current_amount >= upper_bound:
            pass
        else:
            for i in range(parallel_cegar):
                print('kek' + str(parallel_dfa))
                print(str(cegar_mode[i]))
                future = p.schedule(
                    start_procedure,
                    args=(
                        input_,
                        examples_,
                        lower_bound,
                        upper_bound,
                        output,
                        sym_breaking,
                        solver,
                        cegar_mode[i],
                        initial_amount,
                        step_amount,
                        assumptions_mode,
                        use_ig,
                        parallel_dfa,
                        q_final,
                        unsat,
                        cond_for_processes,
                        state_amount,
                        used_colour,
                        parallel_solver_path,
                        parallel_solver_file
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
            parallel_dfa,
            q_final,
            unsat,
            cond_for_processes,
            state_amount,
            used_colour,
            parallel_solver_path,
            parallel_solver_file
        )
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
        amountIter: int,
        q_final,
        unsat,
        cond_for_processes,
        state_amount,
        used_colour,
        parallel_solver_path,
        parallel_solver_file
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
                    assumptions_mode,
                    used_colour,
                    parallel_solver_path,
                    parallel_solver_file)
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

