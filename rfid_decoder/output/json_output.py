from __future__ import annotations

import json


def result_to_json(result: dict) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
