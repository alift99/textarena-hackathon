import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from system_prompts import STANDARD_GAME_PROMPT
from textarena.core import Agent

# Load API key from .env
load_dotenv()

conversation_history = []


class OpenAILangchainAgent(Agent):
    """Agent class using the OpenAI API with LangChain tool_calling_agent to generate responses."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        tools: List = [],
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the OpenAI agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs

        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        self.model = ChatOpenAI(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.tools = tools

        agent = create_tool_calling_agent(self.model, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def _make_request(self, observation: str) -> str:
        """
        Make a single API request to OpenAI and return the generated message.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response text.
        """

        conversation_history.append(("input", observation))

        # Make the API call using the provided model and messages.
        response = self.agent_executor.invoke({"input": observation})

        conversation_history.append(("output", response["output"]))

        with open("history.json", "w") as file:
            json.dump(conversation_history, file, indent=4)

        return response["output"]

    def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenAI API and return the generated response.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(
                f"Observation must be a string. Received type: {type(observation)}"
            )
        return self._retry_request(observation)


class ClaudeLangchainAgent(Agent):
    """Agent class using the Anthropic API with LangChain tool_calling_agent to generate responses."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        tools: List = [],
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the Claude agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs

        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

        self.model = ChatAnthropic(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.tools = tools

        agent = create_tool_calling_agent(self.model, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def _make_request(self, observation: str) -> str:
        """
        Make a single API request to OpenAI and return the generated message.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response text.
        """

        conversation_history.append({"input": observation})

        # Make the API call using the provided model and messages.
        response = self.agent_executor.invoke({"input": observation})
        conversation_history.append({"output": response["output"][0]["text"]})

        json.dump(conversation_history, open("history.json", "w"), indent=4)

        return response["output"][0]["text"]

    def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenAI API and return the generated response.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(
                f"Observation must be a string. Received type: {type(observation)}"
            )
        return self._retry_request(observation)


class ClaudeLangchainAgentCustom(Agent):
    """Agent class using the Anthropic API with LangChain tool_calling_agent to generate responses."""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = STANDARD_GAME_PROMPT,
        tools: List = [],
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the Claude agent.

        Args:
            model_name (str): The name of the model.
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            verbose (bool): If True, additional debug info will be printed.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.kwargs = kwargs

        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

        self.model = ChatAnthropic(model=model_name)
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
                # Placeholders fill up a **list** of messages
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.tools = tools

        agent = create_tool_calling_agent(self.model, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)
        self.id = "Self"

    def _make_request(self, observation: str) -> str:
        """
        Make a single API request to OpenAI and return the generated message.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response text.
        """

        INST = """"""

        # INST = """Your goal is increase the total value of your resources. This means you should always act in your own favour. Please think and plan before answering. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You must use this format and do not write anything outside of these tags. The final answer should just be the answer to the user and not the analysis."""

        # INST = """The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You must use this format and do not write anything outside of these tags. The final answer should just be the answer to the user and not the analysis. The final answer should follow the given instructions below."""

        # INST = """You are a competitive game player. Please read the game instructions carefully, and always follow the required format. You output should be structured as follows: the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You must use this format and do not write anything outside of these tags. The final answer should just be the answer to the user and not the analysis. The final answer should follow the given instructions below."""

        # INST = """Strategy tips: You should try to extract the values of each item from your opponent and use to your advantage. By default, you should reveal as little information as possible to your opponent, The exception is when you will reveal information to deceive your opponent to your advantage."""

        # Edit the instructions
        obs_ls = observation.split("\n\n")
        obs_ls[0] = obs_ls[0] + INST
        observation = "\n\n".join(obs_ls)

        conversation_history.append({"input": observation})

        # Make the API call using the provided model and messages.
        response = self.agent_executor.invoke({"input": observation})

        # Process output
        output = response["output"][0]["text"]
        reasoning = output.split("</think>\n<answer>")[0]
        answer = output.split("</think>\n<answer>")[-1]

        conversation_history.append(
            {"output": output, "reasoning": reasoning, "answer": answer}
        )
        json.dump(conversation_history, open("baseline_history.json", "w"), indent=4)

        return answer

    def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    time.sleep(delay)
        raise last_exception

    def __call__(self, observation: str) -> str:
        """
        Process the observation using the OpenAI API and return the generated response.

        Args:
            observation (str): The input string to process.

        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(
                f"Observation must be a string. Received type: {type(observation)}"
            )
        return self._retry_request(observation)
