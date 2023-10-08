#!/usr/bin/env python3
# type: ignore

import json
import os
import re
import subprocess
import time

SLEEP = 1.5
# set the URLs of each website, we use the demo sites as an example
os.environ[
    "SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site
print("Done setting up URLs")

# First, run `python scripts/generate_test_data.py` to generate the config files
start_time = time.time()
p = subprocess.run(
    ["python", "scripts/generate_test_data.py"], capture_output=True
)
end_time = time.time()
print(f"Time taken to generate test data: {end_time - start_time} seconds")

# It will generate individual config file for each test example in config_files
assert os.path.exists("config_files/0.json")

# Make sure the URLs in the config files are replaced properly
with open("config_files/0.json", "r") as f:
    config = json.load(f)
    assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (
        os.environ["SHOPPING_ADMIN"],
        config["start_url"],
    )

print("Done generating config files with the correct URLs")

# run bash prepare.sh to save all account cookies, this only needs to be done once
start_time = time.time()
# subprocess.run(["bash", "prepare.sh"])
end_time = time.time()
print(f"Time taken to save account cookies: {end_time - start_time} seconds")
print("Done saving account cookies")

# Init an environment
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from evaluation_harness.evaluators import evaluator_router

# Init the environment
env = ScriptBrowserEnv(
    headless=False,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)

# example 156 as an example
config_file = "config_files/156.json"
# maintain a trajectory
trajectory: Trajectory = []

# set the environment for the current example
start_time = time.time()
obs, info = env.reset(options={"config_file": config_file})
end_time = time.time()
print(f"Time taken to reset environment: {end_time - start_time} seconds")
actree_obs = obs["text"]
print(actree_obs)

# You should see some output like this:
"""
[4] RootWebArea 'Projects · Dashboard · GitLab' focused: True
        [12] link 'Skip to content'
        [28] link 'Dashboard'
        [2266] button '' hasPopup: menu expanded: False
        [63] textbox 'Search GitLab' required: False
        [61] generic 'Use the shortcut key <kbd>/</kbd> to start a search'
        [79] link 'Create new...'
        [95] link 'Issues'
                [97] generic '13 assigned issues'
        [101] link 'Merge requests'
                [104] generic '8 merge requests'"""

# save the state info to the trajectory
state_info: StateInfo = {"observation": obs, "info": info}
trajectory.append(state_info)


# Agent workflow
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    action2str,
    create_id_based_action,
    create_stop_action,
#  click, stop actions

# Add the action to the trajectory
trajectory.append(click_action)


# Now let's try to perform the action of clicking the "Merge request" link
# As the element ID is dynamic each time, we use regex to match the element as the demo
# match = re.search(r"\[(\d+)\] link 'Merge requests'", actree_obs).group(1)
# Create the action click [ELEMENT_ID]
# click_action = create_id_based_action(f"click [{match}]")


# Step and get the new observation
start_time = time.time()
obs, _, terminated, _, info = env.step(click_action)
end_time = time.time()
print(f"Time taken to step and get new observation: {end_time - start_time} seconds")
# New observation
actree_obs = obs["text"]
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# Next click "assign to you"
match = re.search(r"\[(\d+)\] link 'Assigned to you", actree_obs).group(1)
click_action = create_id_based_action(f"click [{match}]")
trajectory.append(click_action)

start_time = time.time()
obs, _, terminated, _, info = env.step(click_action)
end_time = time.time()
print(f"Time taken to step and get new observation: {end_time - start_time} seconds")
actree_obs = obs["text"]
print(actree_obs)
time.sleep(SLEEP)
state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# add a stop action to mark the end of the trajectory
trajectory.append(create_stop_action(""))


# Demo evaluation
start_time = time.time()
evaluator = evaluator_router(config_file)
score = evaluator(
    trajectory=trajectory,
    config_file=config_file,
    page=env.page,
    client=env.get_page_client(env.page),
)
end_time = time.time()
print(f"Time taken for evaluation: {end_time - start_time} seconds")

# as we manually perform the task, the task should be judged as correct
assert score == 1.0
