import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Union, Optional, Any

class BaseEvaluator(metaclass=ABCMeta):
    def __init__(self):
        self._dataset_meta=None
        self.results: List[Any]=[]

    @property
    def dataset_meta(self):
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta):
        self._dataset_meta=dataset_meta

    @abstractmethod
    def process(self, data_samples:list):
        """

        """

    @abstractmethod
    def compute_metrics(self, results:list) -> Any:
        """
        compute metrics
        """

    def reset(self):
        self.results = []


    def evaluate(self, size: int):
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.reuslts`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` methods.'
            )
        results = self.results[:size]
        results = _to_cpu(results)
        metrics = self.compute_metrics(results)

        return metrics


def _to_cpu(data: Any) -> Any:
    """
    transfer all tensors and BaseDataElement to cpu.
    """
    if hasattr(data,'to'):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return (_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k,v in data.items()}
    else:
        return data

