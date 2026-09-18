"""Small deterministic application regression set, not a standardized capability benchmark."""

from __future__ import annotations

import json
from typing import Any


def regression_tasks() -> list[dict[str, Any]]:
    tasks = []
    for i in range(16):
        a, b = 17 + i * 3, 8 + i * 2
        tasks.append(
            {
                "id": f"arithmetic_{i}",
                "category": "arithmetic",
                "prompt": f"Compute ({a} * {b}) + 13. Reply with only the integer answer.",
                "expected": str(a * b + 13),
            }
        )
        expected = {"city": ["Beijing", "London", "Paris", "Tokyo"][i % 4], "days": i + 1}
        tasks.append(
            {
                "id": f"json_{i}",
                "category": "json_extraction",
                "prompt": (
                    "Extract a JSON object with exactly the keys city (string) and days (integer). "
                    "Output JSON only, without markdown or explanation. "
                    f"Booking: city={expected['city']}; duration={expected['days']} days."
                ),
                "expected": expected,
            }
        )
    for i in range(8):
        # Repeated, irrelevant facts intentionally isolate key retrieval, not general QA.
        facts = [
            f"Record {j}: ordinary maintenance completed; no access code is recorded here."
            for j in range(650)
        ]
        code = str(420013 + 7919 * i)
        position = (i * 83) % len(facts)
        facts.insert(position, f"IMPORTANT: the access code for cabinet OMEGA is {code}.")
        tasks.append(
            {
                "id": f"retrieval_{i}",
                "category": "long_retrieval",
                "prompt": "\n".join(facts)
                + "\nWhat is the access code for cabinet OMEGA? Reply with the code only.",
                "expected": code,
                "fact_position_fraction": position / 650,
            }
        )
    return tasks


def judge(task: dict[str, Any], output: str) -> bool:
    if task["category"] == "json_extraction":
        try:
            value = json.loads(output.strip())
        except json.JSONDecodeError:
            return False
        return (
            isinstance(value, dict) and value == task["expected"] and type(value.get("days")) is int
        )
    return output.strip() == task["expected"]
