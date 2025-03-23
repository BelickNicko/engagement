import numpy as np


class AlarmElement:
    def __init__(
        self,
        source: str,
        anomalies: list,
        image: np.ndarray,
        timestamp: int | float,
    ) -> None:
        self.source = source
        self.anomalies = anomalies
        self.image = image
        self.timestamp = timestamp
