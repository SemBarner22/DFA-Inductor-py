import logging
import multiprocessing
import os
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


def run_solver(solver):
    """
        Run a single solver on a given formula.
    """
    logging.info(f"starting {solver}")
    res = solver.solve()
    logging.info(f"finished {solver} -- {res} outcome")


class ParallelSolverPortfolio(SolverWrapper):

    def __init__(self, name):
        super().__init__(name)
        self.solvers = []

    def add_solver(self, solver):
        self.solvers.append(Solver(solver))

    def solve(self, assumptions):
        logging.info("Parallel solving started")
        logging.info("Creating tasks")

        if __name__ == '__main__':
            threads = [multiprocessing.Process(target=run_solver, args=solver) for solver in self.solvers]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()  # waits for thread to complete its task

            logging.info("Main Ended")
        else:
            logging.info("Name is not main")

    def append_formula(self, formula):
        for solver in self.solvers:
            solver.append_formula(formula)

    def add_clause(self, clause):
        for solver in self.solvers:
            solver.add_clause(clause)

    def get_model(self):
        for solver in self.solvers:
            solver.get_model()


class ParallelSolverPathToFile(SolverWrapper):

    def get_model(self):
        log_info(self.result)
        if self.result:

                self.answer = self.answer[2:]
                res = [int(x) for x in self.answer.split(' ')]
                log_info(str(res))
                return res
        else:
            print("No answer")

    def write_to_file(self):
        file = open("dfainductor/parallel/inputDKA.cnf", "w+", encoding="utf8")
        file.write("c A sample .cnf file\n")
        # log_info("c A sample .cnf file\n")
        file.write("p cnf " + str(self.amount_of_variables) + " " + str(len(self.list_of_clauses)) + "\n")
        # log_info("p cnf " + str(self.amount_of_variables) + " " + str(self.list_of_clauses) + "\n")

        for clause in self.list_of_clauses:
            file.write(str(len(clause)) + " " + " ".join(str(x) for x in clause) + " 0" + " \n")


        # converted_list = [str(element) for clause in self.list_of_clauses for element in clause]
        # converted_list_of_list = [str]

        # file.write(",".join(converted_list) + "\n")

        # log_info(",".join(converted_list) + "\n")

    @staticmethod
    def execute():
        # input - аргументы к испольняемому запросу
        exit_code = subprocess.run(['./dfainductor/parallel/starexec_run_Version_1.sh', 'input33.cnf'], shell=True, capture_output=True, text=True)
        # exit = subprocess.Popen(['./dfainductor/parallel/starexec_run_Version_1.sh', 'input33.cnf'], stdout=subprocess.PIPE)
        # exit_code = exit.communicate()
        # res = exit_code.stdout.decode('utf-8')
        res = exit_code
        result = res.stdout.split("\n")[-3]
        # log_info(res.split("\n")[-3])
        log_info(res.stdout.split("\n")[-2])
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
        self.amount_of_variables = max(self.amount_of_variables, len(clause))
        self.list_of_clauses.append(clause)

    def append_formula(self, formula):
        for clause in formula:
            self.add_clause(clause)
