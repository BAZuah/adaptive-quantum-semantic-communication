from dataclasses import dataclass


@dataclass
class TwoNodeTopology:
    source: str = "Node_A"
    destination: str = "Node_B"
    distance: float = 1000.0
    attenuation: float = 0.0002
    num_trials: int = 20
