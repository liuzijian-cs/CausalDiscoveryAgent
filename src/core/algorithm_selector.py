
from .state import GlobalState


class AlgorithmSelector(object):
    def __init__(self) -> None:
        pass

    def forward(self, state: GlobalState):
        if state.algorithm.selected_algorithm is None:
            return state

    def _algorithm_selection(self, state: GlobalState):
        
