#!/usr/bin/env python

from lerobot.robots.dual_scorpion_follower import DualScorpionFollowerConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_setup_motors import COMPATIBLE_DEVICES, SetupConfig
from lerobot.teleoperators.dual_scorpion_leader import DualScorpionLeaderConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


def test_setup_motors_accepts_dual_scorpion_arm_selector():
    cfg = SetupConfig(
        robot=DualScorpionFollowerConfig(right_arm_port="/dev/right", left_arm_port="/dev/left"),
        arm="left",
    )

    assert cfg.device.type == "dual_scorpion_follower"
    assert "dual_scorpion_follower" in COMPATIBLE_DEVICES
    assert "dual_scorpion_leader" in COMPATIBLE_DEVICES


def test_dual_scorpion_factories_are_registered(monkeypatch):
    class RobotSentinel:
        def __init__(self, config):
            self.config = config

    class TeleopSentinel:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr("lerobot.robots.dual_scorpion_follower.DualScorpionFollower", RobotSentinel)
    monkeypatch.setattr("lerobot.teleoperators.dual_scorpion_leader.DualScorpionLeader", TeleopSentinel)

    robot_config = DualScorpionFollowerConfig(right_arm_port="/dev/right", left_arm_port="/dev/left")
    teleop_config = DualScorpionLeaderConfig(right_arm_port="/dev/right", left_arm_port="/dev/left")

    assert isinstance(make_robot_from_config(robot_config), RobotSentinel)
    assert isinstance(make_teleoperator_from_config(teleop_config), TeleopSentinel)
