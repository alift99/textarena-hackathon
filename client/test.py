import os
from dotenv import load_dotenv

import textarena as ta

from langchain_agent import OpenAILangchainAgent


# Load API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize agents
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    0: ta.agents.OpenAIAgent(model_name="gpt-4o-mini"),
    1: ta.agents.OpenAIAgent(model_name="gpt-4o-mini"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SpellingBee-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Player 1", 1: "Player 2"},
)

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
