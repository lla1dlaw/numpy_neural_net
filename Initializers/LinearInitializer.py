import numpy as np


class Initializer:
    def _he(self, in_size: int, out_size: int) -> np.ndarray:
        std_dev = np.sqrt(2.0 / in_size)
        weights = np.random.normal(loc=0.0, scale=std_dev, size=(in_size, out_size))
        return weights
    

    def __init__(self):
        self._dispatcher = {
                    "he": _he,
                }


    def generate(self, method: str, in_size: int, out_size: int) -> np.ndarray:
        try:
            weights = self._dispatcher[method](in_size, out_size)
            return weights
        except KeyError:
            raise KeyError(f"{method} is not a known initialization method.")
            exit()



