# Qyrus-Test

## Setup Instructions
1. **Install Dependencies**  
   Run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add OpenAI API Key**  
   Open the `main.py` file and add your OpenAI API key:
   ```python
   open_ai_key = "sk-<your-api-key>"
   ```

## Running the Application
To run the Python script:
```bash
python3 main.py
```

## Sample Input

### **User Prompts:**
The program will ask for:

- **CSV File Path:**
  ```text
  Enter the path of your CSV file: /path/to/your_file.csv
  ```

- **Query:**
  ```text
  Enter your query: Which department has the highest average salary?
  ```

## Sample Output

### **Generated Python Code:**
```text
=== Generated Python Code ===

# Import pandas library
import pandas as pd

# Load CSV data into a DataFrame
df = pd.read_csv('data.csv')

# Clean missing or invalid data
df = df.dropna()

# Convert categorical columns to numeric values
df['Bonus Eligible'] = df['Bonus Eligible'].map({'Yes': 1, 'No': 0})

# Group by department and calculate the average salary
avg_salary = df.groupby('Department')['Salary'].mean()

# Find the department with the highest average salary
highest_salary = avg_salary.idxmax()

# Print the result
print("The department with the highest average salary is:", highest_salary)
=============================

=== Code Execution Output ===
The department with the highest average salary is: Support
```
# **Approach**

This Python script takes a query from the user and dynamically generates executable Python code to answer the query using **LangChain**, **OpenAI embeddings**, and **Chroma** for document retrieval.

---

## **1. User Input Collection and Validation**
- The program first collects the **CSV file path** and **user query**.
- Ensures that the **user-provided input** is valid before processing further.

---

## **2. Document Preparation and Splitting**
- Converts the CSV data into small, readable text chunks for embedding.
- **Steps:**
  - Load the CSV data as a "document" using **`CSVLoader`**.
  - Split the document into smaller chunks using **`RecursiveCharacterTextSplitter`** ().
  - **Schema Information:** Adds data schema (data types) and the **first 5 sample rows** for context.

---

## **3. Embedding Generation**
- Generates embeddings for the text chunks using **OpenAI's `text-embedding-ada-002`** model.
- **Purpose:**
  - Embeddings help convert text (both CSV chunks and the user query) into **numerical vectors**.
  - Enables **similarity-based retrieval** of relevant CSV chunks.

---

## **4. Vector Store (Chroma) for Document Retrieval**
- Stores and retrieves the most relevant text chunks based on the user query.
- **Steps:**
  - Store the document embeddings using **Chroma** (a local vector store).
  - Retrieve the **most relevant document chunks** using `retriever.get_relevant_documents(user_query)`.

---

## **5. Custom Prompt for OpenAI Model**
- Formulates a structured prompt for the OpenAI model.
- **Prompt Components:**
  - **Context:** Schema information, sample rows, and the retrieved document chunks.
  - **User Query:** The question asked by the user.
  - **Instructions:** Provides detailed guidelines for generating the Python code snippet.

---

## **6. Code Generation and Execution**
- Generates a Python code snippet to answer the query.
- Executes the generated Python code to produce the final result.
