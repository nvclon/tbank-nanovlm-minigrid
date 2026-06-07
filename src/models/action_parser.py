ACTION_NAME_TO_ID = {
    "left": 0,
    "right": 1,
    "forward": 2,
}

VALID_ACTIONS = set(ACTION_NAME_TO_ID.keys())


def parse_action_only(text: str):
    text = text.strip().lower()

    for action in ["forward", "left", "right"]:
        if text == action:
            return ACTION_NAME_TO_ID[action], False

    for action in ["forward", "left", "right"]:
        if action in text:
            return ACTION_NAME_TO_ID[action], False

    return ACTION_NAME_TO_ID["forward"], True


def parse_text_action(text: str):
    lines = text.strip().lower().splitlines()

    for line in reversed(lines):
        line = line.strip()

        if line.startswith("action:"):
            action_text = line.replace("action:", "").strip()

            for action in ["forward", "left", "right"]:
                if action_text == action:
                    return ACTION_NAME_TO_ID[action], False

            for action in ["forward", "left", "right"]:
                if action in action_text:
                    return ACTION_NAME_TO_ID[action], False

    return ACTION_NAME_TO_ID["forward"], True