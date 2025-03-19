import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from langchain_openai import ChatOpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate

# Simple list of prompts to test
PROMPTS = [
    "Write a one-sentence description of a sunny day.",
    "What are three benefits of exercise?",
    "Explain the concept of machine learning briefly.",
    "Write a short haiku about technology.",
    "List three healthy breakfast ideas.",
    "What is the capital of France?",
    "Explain how a refrigerator works in one sentence.",
    "Write a short joke about programming.",
    "What are the primary colors?",
    "Describe the taste of chocolate.",
]
MAX_TOKENS = 100
TEMPERATURE = 0
MODEL_NAME = "gpt-4o-mini"

# Initialize clients
openai_client = OpenAI()
langchain_llm = LangChainOpenAI(
    temperature=TEMPERATURE, max_tokens=MAX_TOKENS, model_name=MODEL_NAME
)


def test_openai_sdk(prompt):
    """Test latency using OpenAI SDK directly"""
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    end_time = time.time()
    return end_time - start_time, response.choices[0].message.content


def test_langchain(prompt):
    """Test latency using LangChain"""
    prompt_template = PromptTemplate(input_variables=["query"], template="{query}")
    chain = prompt_template | langchain_llm

    start_time = time.time()
    response = chain.invoke(prompt)
    end_time = time.time()
    return end_time - start_time, response


def run_tests(num_runs=3):
    """Run latency tests for both methods multiple times and collect results"""
    results = {"prompt": [], "method": [], "latency": [], "response": []}

    for prompt in PROMPTS:
        print(f"Testing prompt: {prompt[:30]}...")

        # Test OpenAI SDK
        sdk_latencies = []
        for _ in range(num_runs):
            latency, response = test_openai_sdk(prompt)
            sdk_latencies.append(latency)
            results["prompt"].append(prompt)
            results["method"].append("OpenAI SDK")
            results["latency"].append(latency)
            results["response"].append(response)
            time.sleep(0.5)  # To avoid rate limiting

        print(f"  OpenAI SDK avg: {statistics.mean(sdk_latencies):.4f}s")

        # Test LangChain
        lc_latencies = []
        for _ in range(num_runs):
            latency, response = test_langchain(prompt)
            lc_latencies.append(latency)
            results["prompt"].append(prompt)
            results["method"].append("LangChain")
            results["latency"].append(latency)
            results["response"].append(response)
            time.sleep(0.5)  # To avoid rate limiting

        print(f"  LangChain avg: {statistics.mean(lc_latencies):.4f}s")
        print()

    return pd.DataFrame(results)


def visualize_results(df):
    """Create visualizations of the latency test results"""
    # Calculate average latency per method
    avg_latency = df.groupby(["method"])["latency"].mean().reset_index()
    print("Average Latency by Method:")
    print(avg_latency)

    # Average latency by prompt and method
    pivot_df = df.pivot_table(
        index="prompt", columns="method", values="latency", aggfunc="mean"
    ).reset_index()

    # Calculate overhead percentage
    if "LangChain" in pivot_df.columns and "OpenAI SDK" in pivot_df.columns:
        pivot_df["Overhead (%)"] = (
            (pivot_df["LangChain"] - pivot_df["OpenAI SDK"]) / pivot_df["OpenAI SDK"]
        ) * 100

    print("\nLatency by Prompt (seconds):")
    pd.set_option("display.max_rows", None)
    print(pivot_df)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_latency.plot(kind="bar", x="method", y="latency", ax=ax)
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Average Latency: OpenAI SDK vs LangChain")
    plt.tight_layout()
    plt.savefig("latency_comparison.png")
    print("Chart saved as 'latency_comparison.png'")

    return pivot_df


if __name__ == "__main__":
    print("Starting latency tests...")
    print(f"Testing {len(PROMPTS)} prompts with multiple runs each.")
    print("-" * 50)

    # Run the tests
    results_df = run_tests(num_runs=5)

    print("-" * 50)
    print("Tests completed! Analyzing results...")

    # Analyze and visualize results
    analysis = visualize_results(results_df)

    # Save results to CSV
    results_df.to_csv("latency_test_results_raw.csv", index=False)
    analysis.to_csv("latency_test_results_analysis.csv", index=False)
    print("Results saved to CSV files.")
