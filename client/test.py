import os

import textarena as ta
from dotenv import load_dotenv
from langchain_agent import (
    ClaudeLangchainAgent,
    ClaudeLangchainAgentCustom,
    OpenAILangchainAgent,
)

# Load API key from .env
load_dotenv()

# Initialize agents
agents = {
    0: ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest"),
    1: ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SimpleNegotiation-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Opponent", 1: "Self"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
