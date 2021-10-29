import logging
import multiprocessing
import subprocess

from pysat.solvers import Solver


class SolverWrapper:
    def __init__(self, name):
        logger_format = '%(asctime)s:%(threadName)s:%(message)s'
        logging.basicConfig(format=logger_format, level=logging.INFO, datefmt="%H:%M:%S")

    def append_formula(self, formula):
        pass

    def add_clause(self, clause):
        pass

    def solve(self):
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

    def solve(self):
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

    def write_to_file(self):
        file = open("../conf.cnf", "w")
        file.write("c A sample .cnf file\n")
        file.write("p cnf " + str(self.amount_of_variables) + " " + str(self.list_of_clauses) + "\n")
        for clause in self.list_of_clauses:
            file.write(",".join(clause) + "\n")

    @staticmethod
    def execute():
        exit_code = subprocess.call('./practice.sh')
        print(exit_code)

    def solve(self):
        self.write_to_file()
        self.execute()

    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.list_of_clauses = []
        self.amount_of_variables = 0

    def add_clause(self, clause):
        self.amount_of_variables = max(self.amount_of_variables, len(clause))
        self.list_of_clauses.append(clause)

    def append_formula(self, formula):
        for clause in formula:
            self.add_clause(clause)
