import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


@pytest.fixture(scope="session")
def robot():
    from a3_supermimic.robot import load_robot_model, RobotLayout
    from a3_supermimic.kinematics.model import ModelKinematics
    model = load_robot_model()
    return model, RobotLayout.from_model(model), ModelKinematics(model, dtype=torch.float64)


@pytest.fixture(scope="session")
def motion(tmp_path_factory):
    from a3_supermimic.scripts.data import make_fixture
    path = tmp_path_factory.mktemp("motions")/"standing.npz"
    make_fixture(path, frames=64)
    return path
