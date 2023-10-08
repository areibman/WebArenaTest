"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import time
from pathlib import Path

import openai
from agentops import Event
from agentops.helpers import get_ISO_time
from llms.providers.openai_utils import ao_client

print("AgentOps client created.")

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"
print("Log file name generated.")

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
print("Logger created and level set to INFO.")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
print("Console handler created and added to logger.")

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
print("File handler created and added to logger.")

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
print("Log format set.")


def config() -> argparse.Namespace:
    print("Configuring parser.")
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default=os.path.join("agent", "prompts", "jsons", "p_cot_id_actree_2s.json"),
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()
    print("Arguments parsed.")

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""
    print("Checking for early stop.")

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
) -> None:
    print("Starting test.")
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )
    print("ScriptBrowserEnv created.")
    import pdb

    pdb.set_trace()
    # Create events for each config file
    for config_file in config_file_list:
        print("CONFIG:" + config_file)
        try:
            init_timestamp = get_ISO_time()
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )
            print("RenderHelper created.")

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = (
                    _c["intent"]
                    + '. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside `````` like this: ```[action]```'
                )
                intent += _c["intent"]
                task_id = _c["task_id"]

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            print("Agent reset.")
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)
            print("Environment reset.")

            meta_data = {"action_history": ["None"]}
            import pdb

            pdb.set_trace()
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                    except ValueError as e:
                        # get the error message
                        import pdb

                        pdb.set_trace()
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)
                print("Action appended to trajectory.")

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)
                print("Action rendered.", action_str)

                if action["action_type"] == ActionTypes.STOP:
                    import pdb

                    pdb.set_trace()
                    print("stopping")
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                print("Environment step taken.")

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            print("Evaluator routed.")
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)
            print("Score appended.")

            if score == 1:
                ao_client.record(
                    Event(
                        f"[Intent]: {intent} [Config file] {config_file}",
                        result="Success",
                        init_timestamp=init_timestamp,
                    )
                )
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                ao_client.record(
                    Event(
                        f"[Intent]: {intent} [Config file] {config_file}",
                        result="Fail",
                        init_timestamp=init_timestamp,
                    )
                )
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")
                print("Trace saved.")

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            pdb.set_trace()
            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format__exc())  # write stack trace to file
            print("Error written to file.")

        render_helper.close()
        print("RenderHelper closed.")

    env.close()
    import numpy as np

    logger.info(f"Average score: {np.mean(scores)}")
    print("Environment closed.")


def prepare(args: argparse.Namespace) -> None:
    print("Preparing.")
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()
    print("Prompt python files converted to json.")

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")
        print("Result directory created.")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)
        print("Traces directory created.")

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")
        print("Log file logged.")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    print("Getting unfinished tasks.")
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    print("Dumping config.")
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")
            print("Config dumped.")


if __name__ == "__main__":
    print("Starting main.")
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)
    print("Preparation done.")

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = True
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)
    print("Config dumped.")

    agent = construct_agent(args)
    print("Agent constructed.")
    test(args, agent, test_file_list)
    print("Test completed.")
