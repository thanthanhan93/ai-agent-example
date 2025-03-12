from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, LiteLLMModel, CodeAgent
import sys

load_dotenv()

import pandas as pd

DOCS_PATH = "huggingface_course/data/fakeuser.csv"

df = pd.read_csv(DOCS_PATH)

# Create text column for embeddings
df["text_column"] = df.apply(
    lambda row: ", ".join([f"{col}: {row[col]}" for col in df.columns]),
    axis=1,
)

# Create documents
docs = [
    Document(
        page_content=doc["text_column"],
        metadata={
            "full_name": doc["Full Name"],
            "email": doc["Email"],
            "phone_number": doc["Phone Number"],
            "location": doc["Location"],
            "country": doc["Country"],
            "company": doc["Company"],
            "role": doc["Role"],
        },
    )
    for doc in df.to_dict(orient="records")
]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs = text_splitter.split_documents(docs)


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=3)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n"
                + "\n".join([f"{key}: {value}" for key, value in doc.metadata.items()])
                for i, doc in enumerate(docs)
            ]
        )


retriever_tool = RetrieverTool(docs)

model = LiteLLMModel(
    model_id="ollama/qwen2.5:7b",
    api_base="http://localhost:11434",
    api_key="ollama",
)

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,
)

agent.prompt_templates["system_prompt"] = (
    agent.prompt_templates["system_prompt"]
    + '\nIf you do not have information from retriever, just say "I cannot found the answer".'
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent_output = agent.run(sys.argv[1])
        print("Final output:")
        print(agent_output)
    else:
        print("Please provide an argument")
