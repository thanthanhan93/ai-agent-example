import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import dotenv
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console

console = Console()
dotenv.load_dotenv()


class EmailState(TypedDict):
    # The email being processed
    email: Dict[str, Any]  # Contains subject, sender, body, etc.

    # Analysis and decisions
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]

    # Response generation
    draft_response: Optional[str]

    # Processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis


class EmailResponse(BaseModel):
    answer: bool = Field(description="determine if this email is spam")
    reason: str = Field(description="The reason for the answer")
    category: str = Field(
        description="categorize email (inquiry, complaint, thank you, etc.)"
    )


# Initialize our LLM
model = ChatOpenAI(temperature=0, model="gpt-4o-mini")


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    # Here we might do some initial preprocessing
    console.print(
        f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}",
        style="cyan",
    )

    # No state changes needed here
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    First, determine if this email is spam or not based on the content.
    If it is spam, explain why in reason.
    If it is not spam, categorize it (inquiry, complaint, thank you, etc.) in category.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    structured_llm = model.with_structured_output(EmailResponse)
    response = structured_llm.invoke(messages)

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # Return state updates
    return {
        "is_spam": response.answer,
        "spam_reason": response.reason,
        "email_category": response.category,
        "messages": new_messages,
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    console.print(
        f"[red]Alfred has marked the email as spam. Reason: {state['spam_reason']}[/red]"
    )
    console.print("[yellow]The email has been moved to the spam folder.[/yellow]")

    # We're done processing this email
    return {}


def draft_email_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content},
    ]

    # Return state updates
    return {"draft_response": response.content, "messages": new_messages}


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    console.print("\n" + "=" * 50, style="cyan")
    console.print(
        f"[bold magenta]Sir,[/bold magenta] you've received an email from [bold green]{email['sender']}[/bold green]."
    )
    console.print(
        f"[bold yellow]Subject:[/bold yellow] [bold blue]{email['subject']}[/bold blue]"
    )
    console.print(f"[bold orange]Category:[/bold orange] {state['email_category']}")
    console.print("\nI've prepared a draft response for your review:", style="cyan")
    console.print("-" * 50, style="cyan")
    console.print(state["draft_response"], style="white")
    console.print("=" * 50 + "\n", style="cyan")

    # We're done processing this email
    return {}


def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_email_response", draft_email_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

email_graph.add_edge(START, "read_email")
# Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {"spam": "handle_spam", "legitimate": "draft_email_response"},
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_email_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# Compile the graph
compiled_graph = email_graph.compile()


# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith",
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100.",
}

# Process the legitimate email
langfuse_handler = CallbackHandler()

# Process legitimate email
legitimate_result = compiled_graph.invoke(
    input={
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": [],
    },
    config={"callbacks": [langfuse_handler]},
)

# Process the spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke(
    {
        "email": spam_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": [],
    },
    config={"callbacks": [langfuse_handler]},
)

# Draw the graph
compiled_graph.get_graph().draw_mermaid_png(output_file_path="resources/graph.png")
