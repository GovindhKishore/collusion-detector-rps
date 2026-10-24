import json
from pathlib import Path

from setup.api_startup import RPSGame
from detection.collusion_detector import build_tables, gamma, detect

RESULT_PATH = Path(__file__).parent / "results.json"

NROUND = 1000

with open(RESULT_PATH, "r") as fh:
    results = json.load(fh)


def main():
    # config A: gemma, gemma, gemma
    configA = RPSGame("gemma", "gemma", "gemma", nround=NROUND, state=False)
    configA_state = RPSGame("gemma", "gemma", "gemma", nround=NROUND, state=True)

    # config B: gemma, gemma, mistral
    configB = RPSGame("gemma", "gemma", "mistral", nround=NROUND, state=False)
    configB_state = RPSGame("gemma", "gemma", "mistral", nround=NROUND, state=True)

    # config C: gemma, mistral, qwen
    configC = RPSGame("gemma", "mistral", "qwen", nround=NROUND, state=False)
    configC_state = RPSGame("gemma", "mistral", "qwen", nround=NROUND, state=True)
    #
    # # RUNNING WITH NO STATE
    # if len(results["gemma,gemma,gemma,False"]) != NROUND:
    #     configA_results = configA.run()
    #     print(f"Got {len(configA_results)} game rounds for config A")
    #     results["gemma,gemma,gemma,False"] = configA_results
    #     with open(RESULT_PATH, "w") as f:
    #         json.dump(results, f)
    #
    # if len(results["gemma,gemma,mistral,False"]) != NROUND:
    #     print("Running CONFIGB")
    #     configB_results = configB.run()
    #     print(f"Got {len(configB_results)} game rounds for config B")
    #     results["gemma,gemma,mistral,False"] = configB_results
    #     with open(RESULT_PATH, "w") as f:
    #         json.dump(results, f)
    #
    # if len(results["gemma,mistral,qwen,False"]) != NROUND:
    #     configC_results = configC.run()
    #     print(f"Got {len(configC_results)} game rounds for config C")
    #     results["gemma,mistral,qwen,False"] = configC_results
    #     with open(RESULT_PATH, "w") as f:
    #         json.dump(results, f)
    #
    # print(configA.agent_A.retries, configA.agent_B.retries, configA.agent_C.retries)
    # print(configB.agent_A.retries, configB.agent_B.retries, configB.agent_C.retries)
    # print(configC.agent_A.retries, configC.agent_B.retries, configC.agent_C.retries)


    # RUNNING WITH STATE
    if len(results["gemma,gemma,gemma,True"]) != NROUND:
        configA_state_results = configA_state.run()
        print(f"Got {len(configA_state_results)} game rounds for config A (state)")
        results["gemma,gemma,gemma,True"] = configA_state_results
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f)

    if len(results["gemma,gemma,mistral,True"]) != NROUND:
        configB_state_results = configB_state.run()
        print(f"Got {len(configB_state_results)} game rounds for config B (state)")
        results["gemma,gemma,mistral,True"] = configB_state_results
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f)

    if len(results["gemma,mistral,qwen,True"]) != NROUND:
        configC_state_results = configC_state.run()
        print(f"Got {len(configC_state_results)} game rounds for config C (state)")
        results["gemma,mistral,qwen,True"] = configC_state_results
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f)

    print(configA_state.agent_A.retries, configA_state.agent_B.retries, configA_state.agent_C.retries)
    print(configB_state.agent_A.retries, configB_state.agent_B.retries, configB_state.agent_C.retries)
    print(configC_state.agent_A.retries, configC_state.agent_B.retries, configC_state.agent_C.retries)


if __name__ == "__main__":
    main()