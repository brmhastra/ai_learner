from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# Create a ChatGroq model
model = ChatGroq(model="llama-3.1-70b-versatile")

#parent prompt
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List down the main features of the product {product}"),
    ]
)


#pro finder
def analyse_pros(features):
    pro_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given the list of features - {features}, analyse the pros from the features"),
        ]
    )
    return pro_prompt


def analyse_cons(features):
    cons_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given the list of features - {features}, analyse the cons from the features"),
        ]
    )
    return cons_prompt


pros_chain = RunnableLambda(lambda x: analyse_pros(x) | model | StrOutputParser())
cons_chain = RunnableLambda(lambda x: analyse_cons(x) | model | StrOutputParser())

def combine(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"



chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"pros" : pros_chain, "cons" : cons_chain})
    | RunnableLambda(lambda x: combine(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product" : "iPhone 11"})
print(result)