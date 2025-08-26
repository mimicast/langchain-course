# main.py
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate # Support prompt templates
from langchain_openai import ChatOpenAI # Support OpenAI chat models
from langchain_ollama import ChatOllama # Support Ollama chat models

# Load environment variables from a .env file
load_dotenv()

# Main function
def main():
    # Print a greeting message
    print("Hello from langchain-course!")
    # Example usage of PromptTemplate and ChatOpenAI
    information = """
    LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.

LangChain's developers highlight the framework's applicability to use-cases including chatbots, retrieval-augmented generation, document summarization, and synthetic data generation."""

    # Define a prompt template for summarizing information about LangChain
    summary_template = """
    Given the information {information} about LangChain, create the next items;
    1. A concise summary of the information in one sentence.
    2. Two interesting facts about LangChain.
    3. A question that would be good to ask about LangChain.
    Format the response as a list with each item on a new line.
    """
    
    # Create a PromptTemplate instance using the defined template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Use GPT-4o-mini model with temperature 0 for deterministic output
    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Use Ollama's gpt-oss model with temperature 0 for deterministic output
    llm = ChatOllama(model="gpt-oss:latest", temperature=0)
    # Format the prompt with the provided information then pass it to the language model
    chain = summary_prompt_template | llm # Chain is a runnable object, we can invoke it with inputs that match the input variables of the first prompt in the chain
    # Invokes the chain with the provided information and gets the response
    response = chain.invoke(input={"information": information})
    print(response.content)
 
# Entry point
if __name__ == "__main__":
    main()
