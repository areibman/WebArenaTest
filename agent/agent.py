import argparse
import json
from typing import Any

import tiktoken
from beartype import beartype
from agentops import Event
from agentops.helpers import get_ISO_time
from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import lm_config
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    ao_client,
)


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        print("Initializing Agent")
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        print("Predicting next action in Agent")
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        print("Resetting Agent")
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        print("Initializing TeacherForcingAgent")
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        print("Setting action set tag in TeacherForcingAgent")
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        print("Setting actions in TeacherForcingAgent")
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                print("Creating action in TeacherForcingAgent")
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(f"Unknown action type {self.action_set_tag}")
            except ActionParsingError as e:
                print("Error parsing action in TeacherForcingAgent")
                ao_client.record(event=Event("Set actions", result="Fail"))
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        print("Predicting next action in TeacherForcingAgent")
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        print("Resetting TeacherForcingAgent")
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
    ) -> None:
        print("Initializing PromptAgent")
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag

    def set_action_set_tag(self, tag: str) -> None:
        print("Setting action set tag in PromptAgent")
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        init_timestamp = get_ISO_time()
        print("Predicting next action in PromptAgent")
        prompt = self.prompt_constructor.construct(trajectory, intent, meta_data)
        lm_config = self.lm_config
        if lm_config.provider == "openai":
            print("Generating response from OpenAI in PromptAgent")
            if lm_config.mode == "chat":
                response = generate_from_openai_chat_completion(
                    messages=prompt,
                    model=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    top_p=lm_config.gen_config["top_p"],
                    context_length=lm_config.gen_config["context_length"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    stop_token=None,
                )
            elif lm_config.mode == "completion":
                response = generate_from_openai_completion(
                    prompt=prompt,
                    engine=lm_config.model,
                    temperature=lm_config.gen_config["temperature"],
                    max_tokens=lm_config.gen_config["max_tokens"],
                    top_p=lm_config.gen_config["top_p"],
                    stop_token=lm_config.gen_config["stop_token"],
                )
            else:
                ao_client.record(
                    event=Event(
                        "Next action", result="Fail", init_timestamp=init_timestamp
                    )
                )
                raise ValueError(f"OpenAI models do not support mode {lm_config.mode}")
        else:
            ao_client.record(
                event=Event("Next action", result="Fail", init_timestamp=init_timestamp)
            )
            raise NotImplementedError(f"Provider {lm_config.provider} not implemented")

        try:
            print("Parsing response in PromptAgent")
            parsed_response = self.prompt_constructor.extract_action(response)
            print("Parsed response in PromptAgent")
            if self.action_set_tag == "id_accessibility_tree":
                print("Creating ID based action in PromptAgent")
                action = create_id_based_action(parsed_response)
            elif self.action_set_tag == "playwright":
                print("Creating Playwright action in PromptAgent")
                action = create_playwright_action(parsed_response)
            else:
                print("Unknown action type in PromptAgent")
                raise ValueError(f"Unknown action type {self.action_set_tag}")

            print("Adding raw prediction to action in PromptAgent", response)
            action["raw_prediction"] = response

        except ActionParsingError as e:
            print("Error parsing action in PromptAgent", e)
            ao_client.record(
                event=Event("Next action", result="Fail", init_timestamp=init_timestamp)
            )
            action = create_none_action()
            action["raw_prediction"] = response

        return action

    def reset(self, test_config_file: str) -> None:
        print("Resetting PromptAgent")
        pass


def construct_llm_config(args: argparse.Namespace) -> lm_config.LMConfig:
    print("Constructing LLM config")
    llm_config = lm_config.LMConfig(
        provider=args.provider, model=args.model, mode=args.mode
    )
    if args.provider == "openai":
        print("Setting OpenAI LLM config")
        llm_config.gen_config["temperature"] = args.temperature
        llm_config.gen_config["top_p"] = args.top_p
        llm_config.gen_config["context_length"] = args.context_length
        llm_config.gen_config["max_tokens"] = args.max_tokens
        llm_config.gen_config["stop_token"] = args.stop_token
        llm_config.gen_config["max_obs_length"] = args.max_obs_length
    else:
        raise NotImplementedError(f"provider {args.provider} not implemented")
    return llm_config


def construct_agent(args: argparse.Namespace) -> Agent:
    print("Constructing agent")
    llm_config = construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        print("Creating TeacherForcingAgent")
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        print("Creating PromptAgent")
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
            print(constructor_type)
        tokenizer = tiktoken.encoding_for_model(llm_config.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
    else:
        raise NotImplementedError(f"agent type {args.agent_type} not implemented")
    return agent
