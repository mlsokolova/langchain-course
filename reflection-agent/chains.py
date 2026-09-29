from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
"""You are a viral twitter influenser grading a tweet.messages="
Generate critique and recommendations for the user's tweet     
Always provide detailed recommendations, 
including requests for the length, virality, style,etc
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
"You are a twitter techie influenser assistant"
"tasked with writing excellent twitter posts."
"generate the best twitter possible for the user's request"
"if the user provides critique,"
"respond with revised version of your previous attempts",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)  

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm