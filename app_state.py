from typing import List
from dataclasses import dataclass, field

@dataclass
class ModelState:
    num_threads: int = 4
    cores: List[int] = field(default_factory=lambda: [2, 3])
    need_reload: bool = False

@dataclass
class AppState:
    model: ModelState = field(default_factory=ModelState)

app_state = AppState()