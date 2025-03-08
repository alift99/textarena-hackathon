import textarena as ta

from dotenv import load_dotenv
from langchain_agent import OpenAILangchainAgent, ClaudeLangchainAgent, ClaudeLangchainAgentCustom
from tools.spelling_bee_tool import find_valid_word
from tools.simple_negotiation_tools import simple_negotiation_tools
from system_prompts import SPELLINGBEE_PROMPT



# Load API key from .env
load_dotenv()

# Initialize agents
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    1: ClaudeLangchainAgentCustom(model_name="claude-3-5-sonnet-latest"),
    0: ClaudeLangchainAgentCustom(model_name="claude-3-5-sonnet-latest", tools=simple_negotiation_tools),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="SimpleNegotiation-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Claude Agent", 1: "Claude"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
