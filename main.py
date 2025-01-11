import os
import pandas as pd
import sys
import io
import traceback
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import OpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
import openai
import textwrap

open_ai_key = "openai-key"

class MyOpenAIEmbedding(Embeddings):
    def __init__(self, openai_api_key):
        self.api_key = openai_api_key
        openai.api_key = openai_api_key

    def embed_documents(self, texts):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]

def execute_user_code(code_str: str, df: pd.DataFrame, csv_path: str) -> str:
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    exec_error = None

    code_str = code_str.replace("'data.csv'", f"'{csv_path}'")

    normalized_code = textwrap.dedent(code_str)

    try:
        sys.stdout = captured_output
        local_vars = {"df": df, "csv_path": csv_path}
        exec(normalized_code, {}, local_vars)
    except Exception as e:
        exec_error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    if exec_error:
        return f"ERROR:\n{exec_error}"
    return captured_output.getvalue().strip()

def main():
    # Take CSV file path and user query as input
    try:
        csv_path = input("Enter the path of your CSV file: ").strip()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' does not exist.")
        
        # 2. Check if the file has a .csv extension
        if not csv_path.lower().endswith('.csv'):
            raise ValueError("The file provided is not a CSV file. Currently only working with .csv file.")
        user_query = input("Enter your query: ").strip()


        try:
            df_full = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError:
            raise ValueError("The CSV file is not formatted correctly.")

        # Load and split documents
        loader = CSVLoader(file_path=csv_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        schema_info = "\n".join([f"{col}: {str(df_full[col].dtype)}" for col in df_full.columns])
        sample_rows = df_full.head(5).to_string(index=False)  # Add 5 sample rows for context

        for doc in split_docs:
            doc.page_content = (
                f"Schema Information:\n{schema_info}\n\n"
                f"Sample Data (first 5 rows):\n{sample_rows}\n\n"
                f"{doc.page_content}"
            )

        # Embedding model
        custom_embedding = MyOpenAIEmbedding(openai_api_key=open_ai_key)

        # Vector store
        vectorstore = Chroma.from_documents(split_docs, embedding=custom_embedding)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # custom prompt
        prompt_template = """
            You are a Python coding assistant. You have access to the following CSV data:

            {context}

            The user wants to do this operation:
            "{question}"

            Write a Python code snippet that:
            1) Uses pandas to load/process the CSV data (assume it's already in a DataFrame called 'df').
            2) Uses the exact column names from the CSV (e.g., 'Department', 'Salary').
            3) Performs the requested operation accurately.
            4) Prints or returns the final result.
            5) Avoids defining a new function. Write the solution inline using the global DataFrame 'df'
            6) Cleans missing or invalid data where appropriate.
            7) Converts categorical columns to appropriate numeric values where necessary (e.g., 1 for 'Yes', 0 for 'No')

        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        llm = OpenAI(temperature=0.0, openai_api_key=open_ai_key)
        code_chain = LLMChain(llm=llm, prompt=prompt)

        retrieved_docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        generated_code = code_chain.run(context=context, question=user_query)

        print("\n=== Generated Python Code ===")
        print(generated_code)
        print("=============================")

        # Use the CSV file to execute generated code
        result = execute_user_code(generated_code, df_full, csv_path)
        print("\n=== Code Execution Output ===")
        print(result)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except KeyError as e:
        print(f"KeyError: {str(e)}. Check that all required columns exist.")
    except openai.error.AuthenticationError:
        print("Error: Invalid OpenAI API key.")
    except ValueError as e:
        print(f"ValueError: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
