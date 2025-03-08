import textarena as ta

from dotenv import load_dotenv
from agents.langchain_agent import OpenAILangchainAgent, ClaudeLangchainAgent
from agents.router_agent import RouterAgent

from tools.spelling_bee_tool import find_valid_word

from tools.simple_negotiation_tools import simple_negotiation_tools
from system_prompts import SPELLINGBEE_PROMPT, STANDARD_GAME_PROMPT, SIMPLE_NEGOTIATION_PROMPT

general_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=STANDARD_GAME_PROMPT, tools=[])
spellingbee_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SPELLINGBEE_PROMPT, tools=[find_valid_word])
simplenegotiation_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SIMPLE_NEGOTIATION_PROMPT, tools=simple_negotiation_tools)

models = [
    {
        'name': 'general_agent',
        'description': 'An agent for general purpose',
        'model': general_agent
    }, 
    {
        'name': 'spellingbee_agent', 
        'description': 'An agent specialized for spellingbee',
        'model': spellingbee_agent
    },
    {   
        'name': 'simplenegotiation_agent',
        'description': 'An agent specialized for simple negotiation',
        'model': simplenegotiation_agent
    }
]

agent = RouterAgent(
    model_name="claude-3-5-sonnet-latest",
    models=models,
)



# Load API key from .env
load_dotenv()

# Initialize agents
agents = {
    # 0: OpenAIReactAgent(model_name="gpt-4o-mini"),
    0: agent,
    1: general_agent,
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
