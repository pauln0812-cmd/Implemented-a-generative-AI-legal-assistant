import os
import operator
from typing import TypedDict, Annotated

import gradio as gr

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from openrouter import OpenRouter


# =========================================================
# 1. DEFINE STATE
# =========================================================
class State(TypedDict):
    messages: Annotated[list, operator.add]
    memory: str


# =========================================================
# 2. LOAD VECTOR DATABASE
# =========================================================
print("📚 Loading vector database...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma(
    persist_directory="legal_db",
    embedding_function=embedding_model
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.55
    }
)

print("✅ Vector DB loaded")



# =========================================================
# 3. INITIALIZE OPENROUTER
# =========================================================


openrouter_client = OpenRouter(api_key="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")

print("✅ OpenRouter initialized")


# =========================================================
# 4. LLM INVOKE FUNCTION
# =========================================================
def llm_invoke(messages_list, model="z-ai/glm-4.5-air:free"):

    formatted_messages = []

    for msg in messages_list:

        if isinstance(msg, SystemMessage):
            role = "system"

        elif isinstance(msg, HumanMessage):
            role = "user"

        elif isinstance(msg, AIMessage):
            role = "assistant"

        else:
            continue

        formatted_messages.append({
            "role": role,
            "content": msg.content
        })

    response = openrouter_client.chat.send(
        model=model,
        messages=formatted_messages
    )

    return response


# =========================================================
# 5. WORD COUNT
# =========================================================
def word_count(text: str) -> int:
    return len(text.split())


# =========================================================
# 6. MEMORY SUMMARIZATION
# =========================================================
def summarize_memory(memory: str) -> str:

    response = llm_invoke([

        SystemMessage(content="""
Summarize the conversation into
3 concise sentences while preserving
important facts and context.
"""),

        HumanMessage(content=memory)

    ])

    summary = response.choices[0].message.content.strip()

    print("🧾 Memory summarized")

    return summary


# =========================================================
# 7. NORMAL CHAT
# =========================================================
def normal_chat(query, memory, messages):

    response = llm_invoke([

        SystemMessage(content=f"""
You are a helpful assistant.

Conversation memory:
{memory}

Answer naturally and briefly.
""")

    ] + messages)

    answer = response.choices[0].message.content

    return answer


# =========================================================
# 8. RAG CHAT
# =========================================================
def rag_chat(query, memory):

    print("📚 Running RAG retrieval...")
    print("Query",query)
    # -----------------------------------------------------
    # Retrieve docs
    # -----------------------------------------------------
    docs = retriever.invoke(query)

    # -----------------------------------------------------
    # No docs found → fallback
    # -----------------------------------------------------
    if not docs:

        print("⚠️ No relevant docs found")

        return None

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------
    context_parts = []

    for i, doc in enumerate(docs):

        context_parts.append(
            f"""
DOCUMENT {i+1}

{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------
    messages = [

        SystemMessage(content="""
You are a legal RAG assistant.

Use ONLY the provided context.

Rules:
- If answer is not in context, say:
  "I don't know based on the provided documents."
- Do not invent laws
- Do not hallucinate
- Quote relevant legal text when possible
- Keep answers concise and accurate
"""),

        HumanMessage(content=f"""
Conversation memory:
{memory}

Context:
{context}

Question:
{query}
""")
    ]

    # -----------------------------------------------------
    # LLM
    # -----------------------------------------------------
    response = llm_invoke(messages)

    answer = response.choices[0].message.content

    print("💬 RAG Answer:", answer)

    return answer


# =========================================================
# 9. MAIN CHAT ROUTER
# =========================================================
def chatbot_node(state: State):

    query = state["messages"][-1].content

    memory = state.get("memory", "")

    print(f"\n👤 User: {query}")

    # -----------------------------------------------------
    # Try RAG first
    # -----------------------------------------------------
    rag_answer = rag_chat(query, memory)

    # -----------------------------------------------------
    # If no docs → normal chat
    # -----------------------------------------------------
    if rag_answer is None:

        print("🤖 Falling back to normal chat")

        answer = normal_chat(
            query=query,
            memory=memory,
            messages=state["messages"]
        )

    else:
        answer = rag_answer

    assistant_message = AIMessage(content=answer)

    # -----------------------------------------------------
    # Update memory
    # -----------------------------------------------------
    new_memory = memory + f"""

User: {query}

Assistant: {answer}
"""

    # -----------------------------------------------------
    # Summarize memory if too large
    # -----------------------------------------------------
    if word_count(new_memory) > 2000:
        new_memory = summarize_memory(new_memory)

    return {
        **state,
        "messages": state["messages"] + [assistant_message],
        "memory": new_memory
    }


# =========================================================
# 10. BUILD GRAPH
# =========================================================
print("⚙️ Building LangGraph...")

graph_builder = StateGraph(State)

graph_builder.add_node(
    "chatbot",
    chatbot_node
)

graph_builder.add_edge(
    START,
    "chatbot"
)

graph_builder.add_edge(
    "chatbot",
    END
)

graph = graph_builder.compile()

print("✅ Graph ready")


# =========================================================
# 11. RESPONSE FUNCTION
# =========================================================
def respond(user_message, chat_history, memory):

    chat_history = chat_history or []
    memory = memory or ""

    # -----------------------------------------------------
    # Convert history to LangChain messages
    # -----------------------------------------------------
    messages = []

    for msg in chat_history:

        if msg["role"] == "user":

            messages.append(
                HumanMessage(content=msg["content"])
            )

        elif msg["role"] == "assistant":

            messages.append(
                AIMessage(content=msg["content"])
            )

    # -----------------------------------------------------
    # Add current user message
    # -----------------------------------------------------
    messages.append(
        HumanMessage(content=user_message)
    )

    # -----------------------------------------------------
    # Invoke graph
    # -----------------------------------------------------
    result = graph.invoke({
        "messages": messages,
        "memory": memory
    })

    answer = result["messages"][-1].content

    new_memory = result.get("memory", memory)

    # -----------------------------------------------------
    # Update Gradio history
    # -----------------------------------------------------
    updated_history = chat_history + [

        {
            "role": "user",
            "content": user_message
        },

        {
            "role": "assistant",
            "content": answer
        }
    ]

    return "", updated_history, new_memory


# =========================================================
# 12. GRADIO UI
# =========================================================
with gr.Blocks(title="Advanced LangGraph RAG Chatbot") as app:

    gr.Markdown("# 🤖 Advanced LangGraph RAG Chatbot")

    chatbot = gr.Chatbot(height=600)

    memory_state = gr.State("")

    with gr.Row():

        text = gr.Textbox(
            placeholder="Ask a legal question...",
            show_label=False,
            scale=8
        )

        send_btn = gr.Button(
            "🚀 Send",
            variant="primary",
            scale=1
        )

        clear_btn = gr.Button(
            "🗑️ Clear",
            variant="stop",
            scale=1
        )

    # -----------------------------------------------------
    # Send button
    # -----------------------------------------------------
    send_btn.click(
        fn=respond,
        inputs=[
            text,
            chatbot,
            memory_state
        ],
        outputs=[
            text,
            chatbot,
            memory_state
        ]
    )

    # -----------------------------------------------------
    # Press enter
    # -----------------------------------------------------
    text.submit(
        fn=respond,
        inputs=[
            text,
            chatbot,
            memory_state
        ],
        outputs=[
            text,
            chatbot,
            memory_state
        ]
    )

    # -----------------------------------------------------
    # Clear chat
    # -----------------------------------------------------
    clear_btn.click(
        fn=lambda: ([], ""),
        inputs=None,
        outputs=[
            chatbot,
            memory_state
        ]
    )


# =========================================================
# 13. RUN APP
# =========================================================
print("🚀 Launching Gradio app...")

app.launch(
    share=False,
    inbrowser=True
)