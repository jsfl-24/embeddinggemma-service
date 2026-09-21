import threading
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


class LiteRTEngine:
    def __init__(
        self,
        model_path: str | Path,
        max_sequence_length: int,
        num_threads: int | None = None,
    ) -> None:
        self._interpreter = Interpreter(
            model_path=str(model_path), num_threads=num_threads
        )
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details:
            raise RuntimeError("model has no input tensors")
        if not output_details:
            raise RuntimeError("model has no output tensors")

        self._input_index = int(input_details[0]["index"])
        self._output_index = int(output_details[0]["index"])

        expected_shape = [1, max_sequence_length]
        actual_shape = list(input_details[0]["shape"])
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"model input shape {actual_shape} does not match expected "
                f"{expected_shape}"
            )
        if input_details[0]["dtype"] != np.int32:
            raise RuntimeError(
                f"model input dtype {input_details[0]['dtype']} is not int32"
            )
        if len(output_details[0]["shape"]) == 0 or output_details[0]["shape"][0] != 1:
            raise RuntimeError("model output is not a [1, N] tensor")

        self._lock = threading.Lock()

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        with self._lock:
            self._interpreter.set_tensor(self._input_index, input_ids)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_index)
            return np.asarray(output, dtype=np.float32)