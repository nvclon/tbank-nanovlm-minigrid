from src.models.action_parser import parse_action_only, parse_text_action


def main():
    examples = [
        "forward",
        "left",
        "right",
        "Action: forward",
        "Observation: something\nPlan: move\nAction: right",
        "I think the best move is left.",
        "hello world",
    ]

    for text in examples:
        print("=" * 60)
        print(text)
        print("action-only:", parse_action_only(text))
        print("text+action:", parse_text_action(text))


if __name__ == "__main__":
    main()