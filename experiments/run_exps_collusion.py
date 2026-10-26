import json
from pathlib import Path

from setup.api_startup import RPSGame

RESULT_PATH = Path(__file__).parent / "collusion_results.json"
RESULT_PATH1 = Path(__file__).parent / "excluded_char_collusion_results.json"

NROUND = 1000

with open(RESULT_PATH, "r") as fh:
    collusion_results = json.load(fh)

with open(RESULT_PATH1, "r") as fh:
    excluded_char_collusion_results = json.load(fh)


def main():
    # P1 honest, P2 primary, P3 assistant in all cases
    configA = RPSGame("gemma", "gemma", "gemma", nround=NROUND, state=True)

    configB = RPSGame("gemma", "gemma", "mistral", nround=NROUND, state=True)

    configC = RPSGame("gemma", "mistral", "qwen", nround=NROUND, state=True)

    if len(collusion_results["gemma,gemma,gemma,True"]) != NROUND:
        configA_results = configA.run_collusion()
        print(f"Got {len(configA_results)} game rounds for config A (collusion)")
        collusion_results["gemma,gemma,gemma,True"] = configA_results
        with open(RESULT_PATH, "w") as f:
            json.dump(collusion_results, f)

    if len(collusion_results["gemma,gemma,mistral,True"]) != NROUND:
        configB_results = configB.run_collusion()
        print(f"Got {len(configB_results)} game rounds for config B (collusion)")
        collusion_results["gemma,gemma,mistral,True"] = configB_results
        with open(RESULT_PATH, "w") as f:
            json.dump(collusion_results, f)

    if len(collusion_results["gemma,mistral,qwen,True"]) != NROUND:
        configC_results = configC.run_collusion()
        print(f"Got {len(configC_results)} game rounds for config C (collusion)")
        collusion_results["gemma,mistral,qwen,True"] = configC_results
        with open(RESULT_PATH, "w") as f:
            json.dump(collusion_results, f)

    print(configA.agent_A.retries, configA.agent_B.retries, configA.agent_C.retries)
    print(configB.agent_A.retries, configB.agent_B.retries, configB.agent_C.retries)
    print(configC.agent_A.retries, configC.agent_B.retries, configC.agent_C.retries)


    print("\n\n\n")


    configA_ex = RPSGame("gemma", "gemma", "gemma", nround=NROUND, state=False)
    configB_ex = RPSGame("gemma", "gemma", "mistral", nround=NROUND, state=False)
    configC_ex = RPSGame("gemma", "mistral", "qwen", nround=NROUND, state=False)

    if len(excluded_char_collusion_results["gemma,gemma,gemma,False"]) != NROUND:
        configA_ex_results = configA_ex.run_excluded_char()
        print(f"Got {len(configA_ex_results)} game rounds for config A (excluded char)")
        excluded_char_collusion_results["gemma,gemma,gemma,False"] = configA_ex_results
        with open(RESULT_PATH1, "w") as f:
            json.dump(excluded_char_collusion_results, f)

    if len(excluded_char_collusion_results["gemma,gemma,mistral,False"]) != NROUND:
        configB_ex_results = configB_ex.run_excluded_char()
        print(f"Got {len(configB_ex_results)} game rounds for config B (excluded char)")
        excluded_char_collusion_results["gemma,gemma,mistral,False"] = configB_ex_results
        with open(RESULT_PATH1, "w") as f:
            json.dump(excluded_char_collusion_results, f)

    if len(excluded_char_collusion_results["gemma,mistral,qwen,False"]) != NROUND:
        print("Running CONFIGC")
        configC_ex_results = configC_ex.run_excluded_char()
        print(f"Got {len(configC_ex_results)} game rounds for config C (excluded char)")
        excluded_char_collusion_results["gemma,mistral,qwen,False"] = configC_ex_results
        with open(RESULT_PATH1, "w") as f:
            json.dump(excluded_char_collusion_results, f)

    print(configA_ex.agent_A.retries, configA_ex.agent_B.retries, configA_ex.agent_C.retries)
    print(configB_ex.agent_A.retries, configB_ex.agent_B.retries, configB_ex.agent_C.retries)
    print(configC_ex.agent_A.retries, configC_ex.agent_B.retries, configC_ex.agent_C.retries)

if __name__ == "__main__":
    main()


