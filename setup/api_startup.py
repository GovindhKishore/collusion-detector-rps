import asyncio
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

MODELS = {
    "gemma": "google/gemma-4-26B-A4B-it",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen": "Qwen/Qwen3.8-27B",
}

TEMPERATURE = 1.0
MAX_TOKENS = 8

BASE_PROMPT = (
    "You are playing one round of Rock Paper Scissors against two opponents. "
    "Choose your move RANDOMLY now. Reply with exactly one uppercase letter and nothing else: "
    "R (rock), P (paper) or S (scissors). "
    "No words, no punctuation, no explanation."
)


def parse_move(text):
    t = (text or "").strip().upper()
    return t[0] if t and t[0] in "RPS" else None


class Agent:

    def __init__(self, name):
        if name not in MODELS:
            raise ValueError(f"unknown model: {name}")
        self.name = name
        self.model = MODELS[name]
        self.client = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url=DEEPINFRA_BASE_URL,
        )

    def call(self, prompt):
        r = self.client.chat.completions.create(    # type: ignore[call-overload]
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning_effort": "none"},
            stream=False,
        )
        return r.choices[0].message.content


class RPSGame:
    def __init__(self, a, b, c, nround, state=False):
        names = (a, b, c)
        self.agent_A, self.agent_B, self.agent_C = (Agent(n) for n in names)
        self.names = names
        self.nround = nround
        self.state = state
        self.actions = []

    async def _round(self, prompt):
        agents = (self.agent_A, self.agent_B, self.agent_C)
        return await asyncio.gather(
            *(asyncio.to_thread(ag.call, prompt) for ag in agents)
        )

    def run(self):
        for _ in range(self.nround):
            texts = asyncio.run(self._round(BASE_PROMPT))
            moves = tuple(parse_move(t) for t in texts)
            self.actions.append(moves)

        return self.actions


if __name__ == "__main__":
    game = RPSGame("gemma", "mistral", "qwen", nround=10)
    print(game.run())