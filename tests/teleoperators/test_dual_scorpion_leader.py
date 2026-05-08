#!/usr/bin/env python

from unittest.mock import MagicMock, patch

import pytest

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
