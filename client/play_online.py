import textarena as ta

from agents.langchain_agent import ClaudeLangchainAgent, ClaudeLangchainAgentCustom
from agents.router_agent import RouterAgent
import nltk
nltk.download('words')
from tools.nim_tool import solve_nim
from tools.spelling_bee_tool import find_valid_word
from tools.poker_odds_tool import poker_odds

from system_prompts import STANDARD_GAME_PROMPT, SPELLINGBEE_PROMPT, POKER_PROMPT, SIMPLE_NEGOTIATION_PROMPT, THINKING_AGENT_PROMPT, DECEPTION_AGENT_PROMPT, NIM_AGENT_PROMPT
 

model_name = "HumanGPT"
model_description = "A mixture of customized agents, orchestrated by a *router* agent"
email = "alifdaffa.main@gmail.com"


general_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=STANDARD_GAME_PROMPT, tools=[])
spellingbee_agent = ClaudeLangchainAgent(model_name="claude-3-5-sonnet-latest", system_prompt=SPELLINGBEE_PROMPT, tools=[find_valid_word])
poker_agent = ClaudeLangchainAgentCustom(model_name="claude-3-7-sonnet-latest", system_prompt=POKER_PROMPT, tools=[poker_odds])
negotiation_agent = ClaudeLangchainAgentCustom(model_name="claude-3-5-sonnet-latest", system_prompt=SIMPLE_NEGOTIATION_PROMPT, tools=[])
thinking_agent = ClaudeLangchainAgent(model_name="claude-3-7-sonnet-latest", system_prompt=THINKING_AGENT_PROMPT, tools=[])
deceptive_agent = ClaudeLangchainAgentCustom(model_name="claude-3-7-sonnet-latest", system_prompt=DECEPTION_AGENT_PROMPT, tools=[])
nim_agent = ClaudeLangchainAgentCustom(model_name="claude-3-7-sonnet-latest", system_prompt=NIM_AGENT_PROMPT, tools=[solve_nim])

models = [
    {
        'name': 'general_agent',
        'description': 'An agent for general purpose',
        'model': general_agent
    }, 
    {
        'name': 'spellingbee_agent', 
        'description': 'An agent equipped with powerful tools specialized for spellingbee.',
        'model': spellingbee_agent
    },
    {
        'name': 'poker_agent',
        'description': 'An agent equipped with powerful tools specialized for poker.',
        'model': poker_agent
    },
    {
        "name": "negotiation_agent",
        "description": "An agent specialized for simple negotiation",
        "model": negotiation_agent,
    },
    {
        'name': 'thinking_agent',
        'description': 'A general agent with improved reasoning through iterative and multi-step thinking.',
        'model': thinking_agent
    },
    {
        'name': 'deceptive_agent',
        'description': 'A general deceptive agent that performs reasoning discreetly and outputs false thoughts to deceive the opponent.',
        'model': deceptive_agent,
    },
     {
        'name': 'nim_agent',
        'description': 'A agent specialized for NIM gamex',
        'model': nim_agent,
    }
]

agent = RouterAgent(
    model_name="claude-3-5-sonnet-latest",
    models=models,
)

n_games = 10

for _ in range(n_games):
    env = ta.make_online(
        env_id=["Nim-v0"], 
        model_name=model_name,
        model_description=model_description,
        email=email
    )
    env = ta.wrappers.LLMObservationWrapper(env=env)

    env.reset(num_players=1)

    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agent(observation)
        done, info = env.step(action=action)

    try:
        rewards = env.close()
    except:
        pass