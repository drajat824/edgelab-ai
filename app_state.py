from typing import List
from dataclasses import dataclass, field

@dataclass
class ModelState:
    thread: int = 1
    core: List[int] = field(default_factory=lambda: [2, 3])
    fps_camera: int = 15
    need_reload: bool = False
    camera_error: str | None = None
    inference_fps: float = 0.0
    forward_pass_ms: float = 0.0
    model: str = "ssdmobilenet.tflite"

@dataclass
class Board:
    board_id: str
    board_name: str
    ground_truth: List[str] = field(default_factory=list)


@dataclass
class GTState:
    boards: List[Board] = field(
        default_factory=lambda: [
            Board(
                board_id="board-1",
                board_name="Board 1",
                ground_truth=[
                    "Hearts_2",
                    "",
                    "",
                    "Club_5",
                    "Club_6",
                    "Spades_7",
                    "Club_8",
                    "Spades_9",
                    "Club_10",
                    "Hearts_J",
                    "Club_Q",
                    "Hearts_K",
                    "Hearts_J",
                    "Club_Q",
                    "Hearts_K",
                ],
            ),
            Board(
                board_id= "board-2",
                board_name= "Board 2",
                ground_truth= [
                    "Spades_2",
                    "Diamond_3",
                    "Spades_4",
                    "Diamond_5",
                    "Spades_6",
                    "Diamond_7",
                    "Spades_8",
                    "Diamond_9",
                    "Spades_10",
                    "Diamond_J",
                    "Spades_Q",
                    "Diamond_K",
                    "Diamond_J",
                    "Spades_Q",
                    "Diamond_K",
                ]
            )
        ]
    )

@dataclass
class AppState:
    model: ModelState = field(default_factory=ModelState)
    gt_state: GTState = field(default_factory=GTState)

app_state = AppState()