import textarena as ta
from dotenv import load_dotenv

model_name = "test-007"
model_description = "test-007"
email = "simshanghong@gmail.com"

load_dotenv()


# Initialize agent
agent = ta.agents.AnthropicAgent(model_name="claude-3-5-sonnet-latest")


env = ta.make_online(
    env_id=["SpellingBee-v0"],
    model_name=model_name,
    model_description=model_description,
    email=email,
)
print(env)
env = ta.wrappers.LLMObservationWrapper(env=env)


env.reset(num_players=1)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, info = env.step(action=action)


rewards = env.close()
