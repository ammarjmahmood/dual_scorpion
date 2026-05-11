#!/usr/bin/env python

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.robots.dual_scorpion_follower import DualScorpionFollower, DualScorpionFollowerConfig


def _make_bus_mock(name: str) -> MagicMock:
    bus = MagicMock(name=name)
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.torque_disabled.side_effect = _dummy_cm
    bus.is_calibrated = True
    return bus


@pytest.fixture
def follower():
    right_bus = _make_bus_mock("right_bus")
    left_bus = _make_bus_mock("left_bus")

    def _bus_side_effect(*_args, **kwargs):
        bus = right_bus if kwargs["port"] == "/dev/right" else left_bus
        bus.motors = kwargs["motors"]
        positions = {motor: idx for idx, motor in enumerate(bus.motors, 1)}
        bus.sync_read.return_value = positions
        bus.sync_write.return_value = None
        bus.write.return_value = None
        bus.disable_torque.return_value = None
        return bus

    with (
        patch(
            "lerobot.robots.dual_scorpion_follower.dual_scorpion_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(DualScorpionFollower, "configure", lambda self: None),
    ):
        cfg = DualScorpionFollowerConfig(right_arm_port="/dev/right", left_arm_port="/dev/left")
        robot = DualScorpionFollower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_action_features_are_prefixed(follower):
    expected = {f"right_{motor}.pos" for motor in follower.right_bus.motors}
    expected.update({f"left_{motor}.pos" for motor in follower.left_bus.motors})

    assert set(follower.action_features) == expected
    assert set(follower.observation_features) == expected


def test_send_action_splits_prefixed_goal_positions(follower):
    follower.connect()

    action = {}
    action.update({f"right_{motor}.pos": idx for idx, motor in enumerate(follower.right_bus.motors, 1)})
    action.update({f"left_{motor}.pos": idx + 10 for idx, motor in enumerate(follower.left_bus.motors, 1)})

    returned = follower.send_action(action)

    assert returned == action
    follower.right_bus.sync_write.assert_called_once_with(
        "Goal_Position", {motor: idx for idx, motor in enumerate(follower.right_bus.motors, 1)}
    )
    follower.left_bus.sync_write.assert_called_once_with(
        "Goal_Position", {motor: idx + 10 for idx, motor in enumerate(follower.left_bus.motors, 1)}
    )


def test_send_action_clamps_against_unprefixed_present_positions():
    right_bus = _make_bus_mock("right_bus")
    left_bus = _make_bus_mock("left_bus")

    def _bus_side_effect(*_args, **kwargs):
        bus = right_bus if kwargs["port"] == "/dev/right" else left_bus
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = dict.fromkeys(bus.motors, 10.0)
        bus.sync_write.return_value = None
        bus.write.return_value = None
        bus.disable_torque.return_value = None
        return bus

    with (
        patch(
            "lerobot.robots.dual_scorpion_follower.dual_scorpion_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(DualScorpionFollower, "configure", lambda self: None),
    ):
        cfg = DualScorpionFollowerConfig(
            right_arm_port="/dev/right",
            left_arm_port="/dev/left",
            max_relative_target=5.0,
        )
        robot = DualScorpionFollower(cfg)
        robot.connect()

        action = {}
        action.update({f"right_{motor}.pos": 30.0 for motor in robot.right_bus.motors})
        action.update({f"left_{motor}.pos": 0.0 for motor in robot.left_bus.motors})

        returned = robot.send_action(action)

        assert set(returned) == set(action)
        robot.right_bus.sync_write.assert_called_once_with(
            "Goal_Position", dict.fromkeys(robot.right_bus.motors, 15.0)
        )
        robot.left_bus.sync_write.assert_called_once_with(
            "Goal_Position", dict.fromkeys(robot.left_bus.motors, 5.0)
        )


def test_calibrate_joints_updates_only_selected_follower_joint(follower):
    follower.calibration = _full_calibration(follower)
    follower.right_bus.set_half_turn_homings.return_value = {"joint6": 701}
    follower.left_bus.set_half_turn_homings.return_value = {"joint6": 702}

    with (
        patch("builtins.input", return_value=""),
        patch.object(DualScorpionFollower, "_save_calibration", lambda self: None),
        patch(
            "lerobot.utils.partial_calibration.record_selected_ranges_with_full_display",
            side_effect=[
                ({"joint6": 1701}, {"joint6": 2701}),
                ({"joint6": 1702}, {"joint6": 2702}),
            ],
        ),
    ):
        follower.calibrate_joints(["joint7"])

    assert follower.calibration["right_joint5"].homing_offset == 100
    assert follower.calibration["right_joint6"] == MotorCalibration(
        id=7, drive_mode=0, homing_offset=701, range_min=1701, range_max=2701
    )
    assert follower.calibration["left_joint6"] == MotorCalibration(
        id=7, drive_mode=0, homing_offset=702, range_min=1702, range_max=2702
    )

    follower.right_bus.set_half_turn_homings.assert_called_once_with(["joint6"])
    follower.left_bus.set_half_turn_homings.assert_called_once_with(["joint6"])
    follower.right_bus.disable_torque.assert_called_once_with()
    follower.left_bus.disable_torque.assert_called_once_with()
    written_right = follower.right_bus.write_calibration.call_args.args[0]
    assert set(written_right) == set(follower.right_bus.motors)
    assert written_right["joint6"].homing_offset == 701


def _full_calibration(follower: DualScorpionFollower) -> dict[str, MotorCalibration]:
    calibration = {}
    for prefix, bus in (("right", follower.right_bus), ("left", follower.left_bus)):
        for motor, motor_config in bus.motors.items():
            calibration[f"{prefix}_{motor}"] = MotorCalibration(
                id=motor_config.id,
                drive_mode=0,
                homing_offset=100,
                range_min=200,
                range_max=300,
            )
    return calibration
