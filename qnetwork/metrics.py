from dataclasses import dataclass


@dataclass
class NetworkMetrics:
    success_rate: float
    average_fidelity: float
    average_delay: float
    entangled_pairs_used: int
    available_pairs: int = 0
    dropped_pairs: int = 0

    @property
    def communication_fidelity(self) -> float:
        """Fc = P(success) * mean pair fidelity on successes."""
        return float(self.success_rate) * float(self.average_fidelity)
