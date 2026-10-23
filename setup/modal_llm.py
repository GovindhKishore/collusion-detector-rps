import asyncio
import modal


MODEL_REVISIONS = {
    "ministral": ("mistralai/Ministral-3-14B-Instruct-2512-BF16", "3cea74c1ebaf5ce5f5a2553de470e2ceab825142"),
    "qwen": ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"),
    "gemma": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
}

app = modal.App("rps-llm")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.32.0")
    .env({"HF_HUB_CACHE": "/cache"})
)

volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

def build_llm(key: str):
    from vllm import LLM

    repo, rev = MODEL_REVISIONS[key]
    kwargs = dict(
        model=repo,
        revision=rev,
        tokenizer_revision=rev,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    if key == "ministral":
        kwargs.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    return LLM(**kwargs)


def run_chat(llm, prompts, temperature, max_tokens, base_seed):
    from vllm import SamplingParams

    messages = [[{"role": "user", "content": p}] for p in prompts]
    params = [
        SamplingParams(temperature=temperature, max_tokens=max_tokens,
                       seed=base_seed + i)
        for i in range(len(prompts))
    ]
    reqouts = llm.chat(messages, params)
    return [r.outputs[0].text.strip() for r in reqouts]


GPU_CONFIG = dict(
    gpu="L40S",
    image=image,
    volumes={"/cache": volume},
    scaledown_window=60,
    timeout=1800,
    max_containers=1,
)


@app.cls(**GPU_CONFIG)
class Ministral:
    @modal.enter()
    def load(self):
        self.llm = build_llm("ministral")

    @modal.method()
    def generate(self, prompts: list[str], temperature: float = 1.0,
                 max_tokens: int = 8, base_seed: int = 0) -> list[str]:
        return run_chat(self.llm, prompts, temperature, max_tokens, base_seed)


@app.cls(**GPU_CONFIG)
class Qwen:
    @modal.enter()
    def load(self):
        self.llm = build_llm("qwen")

    @modal.method()
    def generate(self, prompts: list[str], temperature: float = 1.0,
                 max_tokens: int = 8, base_seed: int = 0) -> list[str]:
        return run_chat(self.llm, prompts, temperature, max_tokens, base_seed)


@app.cls(**GPU_CONFIG)
class Gemma:
    @modal.enter()
    def load(self):
        self.llm = build_llm("gemma")

    @modal.method()
    def generate(self, prompts: list[str], temperature: float = 1.0,
                 max_tokens: int = 8, base_seed: int = 0) -> list[str]:
        return run_chat(self.llm, prompts, temperature, max_tokens, base_seed)


@app.local_entrypoint()
def main():
    async def run():
        prompt = (
            "Play rock-paper-scissors. Reply with exactly one character and nothing else: "
            "R for Rock, P for Paper, or S for Scissors. Use a capital letter."
        )
        models = {"gemma": Gemma(), "qwen": Qwen(), "ministral": Ministral()}
        calls = [m.generate.remote.aio([prompt] * 4) for m in models.values()]
        results = await asyncio.gather(*calls)
        for name, r in zip(models, results):
            print(name, r)

    asyncio.run(run())