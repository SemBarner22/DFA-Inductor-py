import logging
import multiprocessing
import queue
import subprocess

from pysat.solvers import Solver

from dfainductor.logging_utils import log_info


class SolverWrapper:
    def __init__(self, name):
        logger_format = '%(asctime)s:%(threadName)s:%(message)s'
        logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

    def nof_vars(self):
        pass

    def nof_clauses(self):
        pass

    def append_formula(self, formula):
        pass

    def add_clause(self, clause):
        pass

    def solve(self, assumptions):
        pass

    def get_model(self):
        pass


# class ParallelSolverPortfolio(SolverWrapper):
#
#     def __init__(self, name):import logging
# import multiprocessing
# import subprocess
#
# from pysat.solvers import Solver

from dfainductor.logging_utils import log_info

import multiprocessing as mp


# p = mp.Pool(3)

def run_solver(i, name, clauses):
    """
        Run a single solver on a given formula.
    """
    solver = Solver(name)
    # print("aassaasdad" + str(clauses[0]))
    # for formula in formulas:
    for clause in clauses:
        # print(str(list(clause)))
        solver.add_clause(list(clause))
    res = solver.solve()
    # for clause in clauses:
    #     solver.add_clause(clause)
    # logging.info(f"starting {solver}")
    # res = solver.solve()
    logging.info(f"finished {solver} -- {res} outcome")
    return (res, solver.get_model())


def quit(i):
    # note: p is visible because it's global in __main__
    # logging.info("Quitting")
    # print("quit" + str(i))
    global ress
    global model
    ress, model = i
    # print(ress)
    # print(str(model))
    global p
    p.terminate()
    p = mp.Pool(6)
    # global b
    # b.wait()
    # print(str(ress))
    q.put("a")
    # self.p.terminate()  # kill all pool workers


p = mp.Pool(6)
# b = multiprocessing.Barrier(2)
q = queue.Queue()
model = []
ress = False
# semaphore = multiprocessing.Semaphore(1)


class ParallelSolverPortfolio(SolverWrapper):

    def __init__(self, name, names):
        super().__init__(name)
        self.solvers = names
        import multiprocessing as mp
        # self.p = mp.Pool(len(self.solvers))
        self.model = None
        self.result = False
        # self.formulas = []
        # self.clauses = []
        self.list_of_clauses = []
        self.amount_of_variables = 0

    #
    # def add_solver(self, solver):
    #     self.solvers.append(Solver(solver))

    def solve(self, assumptions):
        logging.info("Parallel solving started")
        logging.info("Creating tasks")

        # if __name__ == '__main__':
        if True:
            results = []
            global p
            # p = mp.Pool(3)
            for i in range(len(self.solvers)):
                # print("started" + str(i))
                r = p.apply_async(run_solver, args=(i, self.solvers[i], self.list_of_clauses
                                                    # str(self.formulas), str(self.clauses)
                                                    ),
                                  callback=quit)
            q.get(block=True)
            global ress
            global model
            self.result = ress
            self.model = model
            logging.info("Main Ended")
        else:
            print("Not main")
        return self.result

    def append_formula(self, formula):
        for clause in formula:
            self.list_of_clauses.append(clause)

    def add_clause(self, clause):
        for c in clause:
            # print(c)
            self.amount_of_variables = max(self.amount_of_variables, abs(c))
        self.list_of_clauses.append(clause)

    def nof_vars(self):
        return self.amount_of_variables

    def nof_clauses(self):
        return len(self.list_of_clauses)

    # return model of ended solver.
    def get_model(self):
        return self.model


class ParallelSolverPathToFile(SolverWrapper):

    def get_model(self):
        r = self.result
        ans = self.answer

        if r:
            # self.answer = "v 1 -2 -3 -4 -5 -6 -7 0"
            ans = ans[2:]
            res = [int(x) for x in ans.split(' ')]
            res.remove(0)
            return res
        else:
            print("No answer")

    def write_to_file(self):

        file = open("inputDKA.cnf", "w+", encoding="utf8")

        file.write("c A sample .cnf file\n")
        # log_info("c A sample .cnf file\n")
        file.write("p cnf " + str(self.amount_of_variables) + " " + str(len(self.list_of_clauses)) + "\n")
        # log_info("p cnf " + str(self.amount_of_variables) + " " + str(self.list_of_clauses) + "\n")

        for clause in self.list_of_clauses:
            file.write(" ".join(str(x) for x in clause) + " 0" + " \n")


        file.close()

        # file.write(",".join(converted_list) + "\n")

        # log_info(",".join(converted_list) + "\n")

    @staticmethod
    def execute():
        # input - аргументы к испольняемому запросу
        exit_code = subprocess.run(['./starexec_run_Version_1.sh'], shell=True, capture_output=True, text=True,
                                   cwd='DFA-Inductor-py/dfainductor/parallel')
        # ./painless-ExMapleLCMDistChronoBT
        # exit = subprocess.Popen(['./dfainductor/parallel/starexec_run_Version_1.sh', 'input33.cnf'], stdout=subprocess.PIPE)
        # print(exit_code)
        # res = exit_code.stdout.decode('utf-8')
        res = exit_code
        result = res.stdout.split("\n")[-3]
        # log_info(res.split("\n")[-3])
        # log_info(res.stdout.split("\n")[-2])

        if result == "s SATISFIABLE":
            print("yes")
            return True, res.stdout.split("\n")[-2]
            # return True, res.stdout.split("\n")[-2]
        else:
            print("no")
            # print(exit_code.stdout.split("\n")[-3])
            return False, None

    def solve(self, assumptions=""):
        self.write_to_file()
        self.result, self.answer = self.execute()
        log_info(str(self.result) + " should be")
        return self.result

    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.list_of_clauses = []
        self.amount_of_variables = 0
        self.result = False
        self.answer = None

    def add_clause(self, clause):
        for c in clause:
            self.amount_of_variables = max(self.amount_of_variables, abs(c))
        self.list_of_clauses.append(clause)

    def append_formula(self, formula):
        for clause in formula:
            self.add_clause(clause)

    def nof_vars(self):
        return self.amount_of_variables

    def nof_clauses(self):
        return len(self.list_of_clauses)
