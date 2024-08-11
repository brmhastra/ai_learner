from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# Load environment variables from .env
# load_dotenv()

# Create a ChatGroq model
model = ChatGroq(model="llama-3.1-70b-versatile")



# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a maths genius"),
        ("human", "Tell me the square of {number}."),
    ]
)

# Define additional processing steps using RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# first_square = RunnableLambda(lambda x: int(x)*int(x))
# second_square = RunnableLambda(lambda x: f"Second Square : {str(int(x)*int(x))}")

# count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words
# chain = prompt_template | model | StrOutputParser() | first_square | second_square

# Run the chain
result = chain.invoke({"number": 3})

# Output
print(result)