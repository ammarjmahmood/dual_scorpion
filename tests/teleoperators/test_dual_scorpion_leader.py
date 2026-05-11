#!/usr/bin/env python

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.teleoperators.dual_scorpion_leader import DualScorpionLeader, DualScorpionLeaderConfig


def _make_bus_mock(name: str) -> MagicMock:
    bus = MagicMock(name=name)
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect():
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.is_calibrated = True
    return bus


@pytest.fixture
def leader():
    right_bus = _make_bus_mock("right_bus")
    left_bus = _make_bus_mock("left_bus")

    def _bus_side_effect(*_args, **kwargs):
        bus = right_bus if kwargs["port"] == "/dev/right" else left_bus
        bus.motors = kwargs["motors"]
        bus.sync_read.return_value = {motor: idx for idx, motor in enumerate(bus.motors, 1)}
        bus.write.return_value = None
        bus.disable_torque.return_value = None
        bus.configure_motors.return_value = None
        return bus

    with patch(
        "lerobot.teleoperators.dual_scorpion_leader.dual_scorpion_leader.FeetechMotorsBus",
        side_effect=_bus_side_effect,
    ):
        cfg = DualScorpionLeaderConfig(right_arm_port="/dev/right", left_arm_port="/dev/left")
        teleop = DualScorpionLeader(cfg)
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_get_action_returns_prefixed_positions(leader):
    leader.connect()

    action = leader.get_action()

    expected = {f"right_{motor}.pos" for motor in leader.right_bus.motors}
    expected.update({f"left_{motor}.pos" for motor in leader.left_bus.motors})
    assert set(action) == expected


def test_action_features_are_prefixed(leader):
    expected = {f"right_{motor}.pos" for motor in leader.right_bus.motors}
    expected.update({f"left_{motor}.pos" for motor in leader.left_bus.motors})

    assert set(leader.action_features) == expected


def test_calibrate_joints_updates_only_selected_leader_arm(leader):
    leader.calibration = _full_calibration(leader)
    leader.left_bus.set_half_turn_homings.return_value = {"joint6": 706}

    with (
        patch("builtins.input", return_value=""),
        patch.object(DualScorpionLeader, "_save_calibration", lambda self: None),
        patch(
            "lerobot.utils.partial_calibration.record_selected_ranges_with_full_display",
            return_value=({"joint6": 1706}, {"joint6": 2706}),
        ),
    ):
        leader.calibrate_joints(["left_joint7"])

    assert leader.calibration["right_joint6"].homing_offset == 100
    assert leader.calibration["left_joint6"] == MotorCalibration(
        id=7, drive_mode=0, homing_offset=706, range_min=1706, range_max=2706
    )

    leader.right_bus.set_half_turn_homings.assert_not_called()
    leader.right_bus.disable_torque.assert_not_called()
    leader.left_bus.set_half_turn_homings.assert_called_once_with(["joint6"])
    leader.left_bus.disable_torque.assert_called_once_with()
    written_left = leader.left_bus.write_calibration.call_args.args[0]
    assert set(written_left) == set(leader.left_bus.motors)
    assert written_left["joint6"].homing_offset == 706


def _full_calibration(leader: DualScorpionLeader) -> dict[str, MotorCalibration]:
    calibration = {}
    for prefix, bus in (("right", leader.right_bus), ("left", leader.left_bus)):
        for motor, motor_config in bus.motors.items():
            calibration[f"{prefix}_{motor}"] = MotorCalibration(
                id=motor_config.id,
                drive_mode=0,
                homing_offset=100,
                range_min=200,
                range_max=300,
            )
    return calibration
