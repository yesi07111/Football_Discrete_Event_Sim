from dataclasses import dataclass, field


@dataclass(order=True)
class SimulationEvent:
    time: float
    counter: int
    event_type: str = field(compare=False)
    team: str = field(compare=False)