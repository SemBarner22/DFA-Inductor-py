from abc import ABC, abstractmethod
from typing import List

from .structures import DFA


class BaseExamplesProvider(ABC):

    def __init__(self, examples: []) -> None:
        self._examples = examples
        self._is_cover = False

    def get_init_examples(self) -> List[str]:
        init_examples = []
        for i in range(min(self._init_examples_size(), len(self._examples))):
            init_examples.append(self._examples[i])
        return init_examples

    def get_all_counter_examples(self, dfa: DFA) -> List[str]:
        counter_examples = []
        it = iter(self._examples)
        try:
            while True:
                word = next(it)
                word_split = word.split()
                if (word_split[0] == '1') != dfa.run(word_split[2:]):
                    counter_examples.append(word)
        except StopIteration:
            pass
        return counter_examples

    def get_counter_examples(self, dfa: DFA) -> List[str]:
        counter_examples = []
        counter_examples_num = self._counter_examples_size()
        it = iter(self._examples)
        try:
            while len(counter_examples) < counter_examples_num:
                word = next(it)
                word_split = word.split()
                if (word_split[0] == '1') != dfa.run(word_split[2:]):
                    counter_examples.append(word)
        except StopIteration:
            pass
        return counter_examples

    def is_provider_calcs_cover(self):
        return self._is_cover

    def get_cover(self, dfa: DFA):
        dfa._perform_cover_calculating(self.examples)
        all_counter_examples = self.get_all_counter_examples(dfa)
        counter_amount = min(len(self.get_counter_examples(dfa)), len(all_counter_examples))
        counter_examples = sorted(all_counter_examples, key=lambda x: dfa.cover_for_word_count(x))[:counter_amount]
        return counter_examples

    def get_all_examples(self) -> List[str]:
        return self._examples

    @abstractmethod
    def _init_examples_size(self) -> int:
        pass

    @abstractmethod
    def _counter_examples_size(self) -> int:
        pass

    @property
    def examples(self):
        return self._examples


class LinearAbsoluteExamplesProvider(BaseExamplesProvider):
    def __init__(self, examples_: [], initial_examples_amount: int, counter_examples_amount: int, is_cover: bool) -> None:
        super().__init__(examples_)
        self._initial_examples_amount = initial_examples_amount
        self._counter_examples_amount = counter_examples_amount
        self._is_cover = is_cover

    def _init_examples_size(self) -> int:
        return self._initial_examples_amount

    def _counter_examples_size(self) -> int:
        return self._counter_examples_amount


class LinearRelativeExamplesProvider(BaseExamplesProvider):
    def __init__(self, examples_: [], initial_examples_amount: int, counter_examples_amount: int, is_cover: bool) -> None:
        super().__init__(examples_)
        self._initial_examples_amount = len(self._examples) // 100 * initial_examples_amount
        self._counter_examples_amount = len(self._examples) // 100 * counter_examples_amount
        self._is_cover = is_cover

    def _init_examples_size(self) -> int:
        return self._initial_examples_amount

    def _counter_examples_size(self) -> int:
        return self._counter_examples_amount


class GeometryProgressionExamplesProvider(BaseExamplesProvider):
    def __init__(self, examples_: [], initial_examples_amount: int, multiplier: int, is_cover: bool) -> None:
        super().__init__(examples_)
        self._initial_examples_amount = initial_examples_amount
        self._counter_examples_amount = initial_examples_amount
        self._multiplier = multiplier
        self._is_cover = is_cover

    def _init_examples_size(self) -> int:
        return self._initial_examples_amount

    def _counter_examples_size(self) -> int:
        self._counter_examples_amount *= self._multiplier
        return self._counter_examples_amount


class NonCegarExamplesProvider(BaseExamplesProvider):
    def _init_examples_size(self) -> int:
        return len(self._examples)

    def _counter_examples_size(self) -> int:
        return 0


def get_examples_provider(examples_: [],
                          cegar_mode: str,
                          initial_examples_amount: int,
                          counter_examples_amount: int) -> BaseExamplesProvider:
    if cegar_mode == 'lin-abs' or 'cover-lin-abs':
        return LinearAbsoluteExamplesProvider(examples_, initial_examples_amount, counter_examples_amount, cegar_mode.startswith('cover'))
    elif cegar_mode == 'rel-abs' or 'cover-rel-abs':
        return LinearRelativeExamplesProvider(examples_, initial_examples_amount, counter_examples_amount, cegar_mode.startswith('cover'))
    elif cegar_mode == 'geom' or 'cover-geom':
        return GeometryProgressionExamplesProvider(examples_, initial_examples_amount, counter_examples_amount, cegar_mode.startswith('cover'))
    else:
        return NonCegarExamplesProvider(examples_)
