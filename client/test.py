import textarena as ta

from langchain_agent import OpenAILangchainAgent, ClaudeLangchainAgent
from tools.spelling_bee_tool import find_valid_word
from system_prompts import SPELLINGBEE_PROMPT

# Initialize agents
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    0: ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SPELLINGBEE_PROMPT, tools=[find_valid_word]),
    1: ta.agents.AnthropicAgent(model_name="claude-3-5-sonnet-latest"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SpellingBee-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Claude Agent", 1: "Claude"},
)

env.reset()
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
