from dataclasses import dataclass


@dataclass(frozen=True)
class HyperparametersConfiguration:
    training_shadow_percentage: float
    testing_shadow_percentage: float
    consensus_percentage: float
    pool_size: int

    def path(self):
        return f'{self.training_shadow_percentage:.2f}_{self.testing_shadow_percentage:.2f}_{self.consensus_percentage:.2f}_{self.pool_size}.bin'
