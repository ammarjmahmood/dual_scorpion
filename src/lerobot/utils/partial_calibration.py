#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2025 S.Satoya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections.abc import Iterable, Sequence
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.utils.utils import enter_pressed, move_cursor_up


def calibration_for_prefixed_bus(
    calibration: dict[str, MotorCalibration],
    prefix: str,
) -> dict[str, MotorCalibration]:
    full_prefix = f"{prefix}_"
    return {
        motor.removeprefix(full_prefix): calib
        for motor, calib in calibration.items()
        if motor.startswith(full_prefix)
    }


def resolve_prefixed_joint_selectors(
    selectors: Sequence[str],
    *,
    motor_names: Iterable[str],
    prefixes: Sequence[str] = ("right", "left"),
) -> dict[str, list[str]]:
    """Resolve selectors like joint7, right_joint6, or left_gripper to per-prefix motor names."""
    known_motors = tuple(motor_names)
    selected = {prefix: [] for prefix in prefixes}

    for selector in selectors:
        selector = selector.strip().lower()
        if not selector:
            continue

        prefix, motor = _split_prefixed_selector(selector, prefixes)
        motor = _resolve_motor_name(motor, known_motors)
        target_prefixes = (prefix,) if prefix else prefixes

        for target_prefix in target_prefixes:
            if motor not in selected[target_prefix]:
                selected[target_prefix].append(motor)

    if not any(selected.values()):
        raise ValueError("No joints were selected for partial calibration.")

    return selected


def recalibrate_prefixed_bus_joints(
    *,
    calibration: dict[str, MotorCalibration],
    prefix: str,
    bus: Any,
    motors: Sequence[str],
    device_label: str,
    position_mode_value: int | None = None,
) -> dict[str, MotorCalibration]:
    """Recalibrate selected bus motors and merge the new values into a prefixed calibration dict."""
    if not motors:
        return calibration

    updated_calibration = calibration.copy()
    missing = [f"{prefix}_{motor}" for motor in motors if f"{prefix}_{motor}" not in updated_calibration]
    if missing:
        raise ValueError(
            "Partial calibration requires an existing calibration entry for every selected joint. "
            f"Missing: {missing}"
        )

    bus.disable_torque()
    if position_mode_value is not None:
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, position_mode_value)

    pretty_motors = ", ".join(motors)
    input(
        f"Move {prefix.upper()} {device_label} {pretty_motors} "
        "to the middle of its range and press ENTER..."
    )
    homing_offsets = bus.set_half_turn_homings(list(motors))

    print(
        f"Move {prefix.upper()} {pretty_motors} through the full range of motion. "
        "Press ENTER when done."
    )
    range_mins, range_maxes = record_selected_ranges_with_full_display(bus, list(motors))

    for motor in motors:
        key = f"{prefix}_{motor}"
        previous = updated_calibration[key]
        updated_calibration[key] = MotorCalibration(
            id=bus.motors[motor].id,
            drive_mode=previous.drive_mode,
            homing_offset=int(homing_offsets[motor]),
            range_min=int(range_mins[motor]),
            range_max=int(range_maxes[motor]),
        )

    bus.write_calibration(calibration_for_prefixed_bus(updated_calibration, prefix))
    return updated_calibration


def record_selected_ranges_with_full_display(
    bus: Any,
    selected_motors: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Mirror MotorsBus.record_ranges_of_motion display for the whole arm, but validate selected motors only.
    """
    display_motors = list(bus.motors)
    selected = set(selected_motors)

    positions = bus.sync_read("Present_Position", display_motors, normalize=False)
    mins = positions.copy()
    maxes = positions.copy()

    user_pressed_enter = False
    while not user_pressed_enter:
        positions = bus.sync_read("Present_Position", display_motors, normalize=False)
        mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
        maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

        print("\n-------------------------------------------")
        print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
        for motor in display_motors:
            marker = "*" if motor in selected else " "
            print(
                f"{marker}{motor:<14} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}"
            )

        if enter_pressed():
            user_pressed_enter = True

        if not user_pressed_enter:
            move_cursor_up(len(display_motors) + 3)

    same_min_max = [motor for motor in selected_motors if mins[motor] == maxes[motor]]
    if same_min_max:
        raise ValueError(
            "Selected motors did not move during partial calibration. "
            f"Check the moving row in the table and retry with the matching --joints value: {same_min_max}"
        )

    return (
        {motor: mins[motor] for motor in selected_motors},
        {motor: maxes[motor] for motor in selected_motors},
    )


def _split_prefixed_selector(selector: str, prefixes: Sequence[str]) -> tuple[str | None, str]:
    for prefix in prefixes:
        full_prefix = f"{prefix}_"
        if selector.startswith(full_prefix):
            return prefix, selector.removeprefix(full_prefix)
    return None, selector


def _resolve_motor_name(selector: str, motor_names: Sequence[str]) -> str:
    if selector in motor_names:
        return selector

    # The Dual Scorpion code uses zero-based joint names. Accept "joint7" as a human-facing alias for
    # the seventh joint, which is stored as "joint6".
    match = re.fullmatch(r"joint(\d+)", selector)
    if match:
        zero_based_name = f"joint{int(match.group(1)) - 1}"
        if zero_based_name in motor_names:
            return zero_based_name

    available = ", ".join(motor_names)
    raise ValueError(f"Unknown joint selector '{selector}'. Available joints: {available}")
