from typing import List, Sequence, Any, Optional, Dict
from dataclasses import dataclass, field
from pydantic import BaseModel

class Point(BaseModel):
    x: float
    y: float
    
@dataclass
class ModelState:
    thread: int = 1
    core: List[int] = field(default_factory=lambda: [2, 3])
    fps_camera: int = 15
    need_reload: bool = False
    camera_error: str | None = None
    inference_fps: float = 0.0
    forward_pass_ms: float = 0.0
    model: str = "ssd-mobilenet.tflite"
    calibration_points: Sequence[Any] = field(default_factory=list)
    latest_evaluation: Optional[Dict[str, Any]] = None

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
                board_name="Board 1 - Dataset",
                ground_truth=[
                    "Diamond_6",
                    "Club_K",
                    "Spades_K",
                    "Club_9",
                    "Diamond_8",
                    "Club_J",
                    "Hearts_2",
                    "Diamond_10",
                    "Club_2",
                    "Diamond_7",
                    "Diamond_K",
                    "Hearts_5",
                    "Club_3",
                    "Hearts_6",
                    "Spades_Q"
                    
                ],
            ),
            Board(
                board_id= "board-2",
                board_name= "Board 2 - Dataset",
                ground_truth= [
                    "Spades_3",
                    "Hearts_J",
                    "Club_Q",
                    "Hearts_8",
                    "Spades_9",
                    "Diamond_4",
                    "Club_6",
                    "Club_5",
                    "Hearts_3",
                    "Hearts_4",
                    "Spades_4",
                    "Diamond_3",
                    "Hearts_Q",
                    "Hearts_9",
                    "Hearts_A"
                ]
            ),
            Board(
                board_id="board-3",
                board_name="Board 3",
                ground_truth=[
                    "Diamond_J",
                    "Club_A",
                    "Spades_5",
                    "Club_8",
                    "Hearts_K",
                    "Hearts_7",
                    "Diamond_2",
                    "Diamond_A",
                    "Club_7",
                    "Spades_7",
                    "Spades_6",
                    "Club_4",
                    "Spades_J",
                    "Diamond_5",
                    "Spades_2"
                ]
            )
        ]
    )
    active_board: str = 'NONE'
    listGT: Sequence[Any] = field(default_factory=list)
    
@dataclass
class AppState:
    model: ModelState = field(default_factory=ModelState)
    gt_state: GTState = field(default_factory=GTState)

app_state = AppState()