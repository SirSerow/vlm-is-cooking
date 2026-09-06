"""Follow a recipe with manual image observations and explicit confirmations."""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .assistant import CookingAssistant
from .observer import OllamaObserver
from .recipe import Recipe, RecipeSession
from .state import Frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    args = parser.parse_args()
    session = RecipeSession(Recipe.load(args.recipe), datetime.now(timezone.utc))
    assistant = CookingAssistant(OllamaObserver(model=args.model), session=session)
    print(session.recipe.title)
    print("Enter an image path to observe, 'confirm' to finish the current step, or 'quit'.")
    while session.current_step is not None:
        recommendation = session.recommend(datetime.now(timezone.utc))
        print(f"\nCurrent: {session.current_step.instruction}")
        print(f"[{recommendation.status}] {recommendation.instruction}\n{recommendation.reason}")
        command = input("> ").strip()
        if command == "quit":
            return
        if command == "confirm":
            session.confirm(datetime.now(timezone.utc))
        elif command:
            step = session.current_step
            if not step.queries:
                print("This step needs your confirmation rather than an image.")
                continue
            path = Path(command.strip('"'))
            # Repeating the same image must not become two independent votes.
            image = path.read_bytes()
            frame = Frame(hashlib.sha256(image).hexdigest(), datetime.now(timezone.utc), image)
            assistant.observe(frame, step.queries)
    print("Recipe checklist complete.")


if __name__ == "__main__":
    main()
