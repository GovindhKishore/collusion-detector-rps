import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import service_tier

load_dotenv()

DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

MODELS = {
    "gemma": "google/gemma-4-26B-A4B-it",
    "mistral": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "qwen": "Qwen/Qwen3.6-27B",
}

TEMPERATURE = 1.0
MAX_TOKENS = 8

BASE_PROMPT = (
    "You are playing one round of Rock Paper Scissors against two opponents. "
    "Choose your move RANDOMLY now. Reply with exactly one uppercase letter and nothing else: "
    "R (rock), P (paper) or S (scissors). "
    "No words, no punctuation, no explanation."
)


def parse_move(text: str | None = None) -> str | None:
    t = (text or "").strip().upper()
    return t[0] if t and t[0] in "RPS" else None


class Agent:

    def __init__(self, name: str) -> None:
        if name not in MODELS:
            raise ValueError(f"unknown model: {name}")
        self.name = name
        self.model = MODELS[name]
        self.client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url=DEEPINFRA_BASE_URL,
        )
        self.retries = 0

    def call(self, prompt: str) -> str | None:
        for attempt in range(3):
            r = self.client.chat.completions.create(    # type: ignore[call-overload]
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning_effort": "none"},
                stream=False,
                timeout=10,

            )
            choice = r.choices[0].message.content
            move = parse_move(choice)
            if move is not None:
                return move
            self.retries += 1

        return None # parse_move(choice) will be None


class RPSGame:
    def __init__(self, a: str, b: str, c: str, nround: int, state: bool=False) -> None:
        names = (a, b, c)
        self.agent_A, self.agent_B, self.agent_C = (Agent(n) for n in names)
        self.names = names
        self.nround = nround
        self.state = state
        self.actions = []

    async def _round(self, prompts: list[str]) -> tuple[str | None, str | None, str | None]:
        return await asyncio.gather(
            asyncio.to_thread(self.agent_A.call, prompts[0]),
            asyncio.to_thread(self.agent_B.call, prompts[1]),
            asyncio.to_thread(self.agent_C.call, prompts[2]),
        )

    def run(self) -> list[tuple[str | None, str | None, str | None]]:
        last_round = []
        for r in range(self.nround):
            if self.state and last_round:
                history = ", ".join(f"Player {i}: {m}" for i, m in enumerate(last_round, 1))
                prompts = [
                    f"You are Player {i}. " + BASE_PROMPT + f" Last round these were the moves: {history}."
                    for i in (1, 2, 3)
                ]
            else:
                prompts = [f"You are Player {i}. " + BASE_PROMPT for i in (1, 2, 3)]

            texts = asyncio.run(self._round(prompts))
            moves = tuple(texts)
            print(f"Round {r+1}: {moves}")
            self.actions.append(moves)
            last_round = moves

        return self.actions



if __name__ == "__main__":
    game = RPSGame("gemma", "mistral", "qwen", nround=10)
    print(game.run())