from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4")

#Session ID store
store = {}

def get_chat_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional life coach. Answer users' questions to help their lives better."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chain = prompt | model | parser
config = {"configurable": {"session_id": "ChatID"}}
with_message_history = RunnableWithMessageHistory(chain, get_chat_history) #args: chain, function to get the chain history
#with_message_history variable will be invoked to get the result.

if __name__ == "__main__":
    while True:
        user_input = input("> ")
        response = with_message_history.invoke( # Instead of .invoke, .stream could be used. Check Langchain https://python.langchain.com/docs/how_to/streaming/#working-with-input-streams
            [HumanMessage(content=user_input)],
            config=config, #Necessary to get the session_id
        )
        print(response)