import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="gpt-4o-mini")
parser_json = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are exper researcher.
Current time: {time}
1. {first_instruction}           
2. Reflect and critique your answer. Be severe to maximize improvement
3. Recommend search queries to reserch information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """
Revise your previous answer using new information.
- you should use previous critique to add important information to your answer.
  - you must include numerical citation in your revised answer to ensure it can be verified
  - add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
  - you should use previous critique to remove superfluous information from your answer and make it no more than 250 symbols.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="""Write about ablation devices in the cardiology.
Write about validation tests of ablation devices on the manufacturing, before it is used in the patient.
What kind of the validation tests are required for the ablation devices to be acceplable by FDA?
        """
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    res = chain.invoke(input={"messages": [human_message]})
    print(res)
