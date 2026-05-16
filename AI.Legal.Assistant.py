import gradio as gr

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openrouter import OpenRouter


# =========================================================
# 1. VECTOR DB
# =========================================================
print("📚 Loading vector DB...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma(
    persist_directory="legal_db",
    embedding_function=embedding_model
)

retriever = db.as_retriever(search_kwargs={"k": 5})

print("✅ Vector DB ready")


# =========================================================
# 2. OPENROUTER
# =========================================================
client = OpenRouter(
    api_key="API Key"
)


# =========================================================
# 3. STREAM LLM
# =========================================================
def stream_llm(messages):

    formatted = []

    for m in messages:

        if hasattr(m, "content"):

            content = m.content

            if not content:
                continue

            if isinstance(m, SystemMessage):
                role = "system"

            elif isinstance(m, HumanMessage):
                role = "user"

            elif isinstance(m, AIMessage):
                role = "assistant"

            else:
                continue

        else:

            role = m.get("role")
            content = m.get("content")

            if not content:
                continue

        formatted.append({
            "role": role,
            "content": str(content)
        })

    return client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=formatted,
        stream=True
    )


# =========================================================
# 4. RAG CONTEXT
# =========================================================
def get_context(query: str):

    docs = retriever.invoke(query)

    if not docs:
        return ""

    return "\n\n".join(
        f"DOC {i+1}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# =========================================================
# 5. CREATE NEW TOPIC
# =========================================================
def create_topic(sessions):

    topic_name = f"New Chat {len(sessions) + 1}"

    sessions[topic_name] = {
        "history": [],
        "memory": ""
    }

    return (
        sessions,
        gr.update(
            choices=list(sessions.keys()),
            value=topic_name
        ),
        []
    )


# =========================================================
# 6. SWITCH TOPIC
# =========================================================
def switch_topic(selected_topic, sessions):

    return sessions[selected_topic]["history"]


# =========================================================
# 7. CHAT FUNCTION
# =========================================================
def chat_stream(
    message,
    history,
    selected_topic,
    sessions
):

    if not selected_topic:
        return "", history, sessions

    current = sessions[selected_topic]

    history = current["history"]
    memory = current["memory"]

    # -----------------------------------------------------
    # RAG CONTEXT
    # -----------------------------------------------------
    context = get_context(message)

    system_prompt = f"""
You are a legal assistant.

Memory:
{memory}

Context:
{context if context else "No relevant documents found."}
"""

    messages = [
        SystemMessage(content=system_prompt)
    ]

    for msg in history:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": message
    })

    # -----------------------------------------------------
    # STREAM RESPONSE
    # -----------------------------------------------------
    response = stream_llm(messages)

    full = ""

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": ""
    })

    yield "", history, sessions

    for chunk in response:

        delta = getattr(
            chunk.choices[0].delta,
            "content",
            ""
        )

        if delta:

            full += delta

            history[-1]["content"] = full

            yield "", history, sessions

    # -----------------------------------------------------
    # SAVE MEMORY
    # -----------------------------------------------------
    memory += f"\nUser: {message}\nAssistant: {full}\n"

    current["history"] = history
    current["memory"] = memory

    sessions[selected_topic] = current

    yield "", history, sessions


# =========================================================
# 8. INITIAL STATE
# =========================================================
default_sessions = {
    "New Chat 1": {
        "history": [],
        "memory": ""
    }
}


# =========================================================
# 9. UI
# =========================================================
with gr.Blocks(
    title="ChatGPT Style RAG",
    theme=gr.themes.Soft()
) as app:

    sessions_state = gr.State(default_sessions)

    with gr.Row():

        # =================================================
        # SIDEBAR
        # =================================================
        with gr.Column(scale=1):

            gr.Markdown("## 💬 Conversations")

            new_chat_btn = gr.Button(
                "➕ New Chat",
                variant="primary"
            )

            topic_list = gr.Radio(
                choices=list(default_sessions.keys()),
                value="New Chat 1",
                show_label=False
            )

        # =================================================
        # MAIN CHAT
        # =================================================
        with gr.Column(scale=4):

            chatbot = gr.Chatbot(
                height=700
            )

            with gr.Row():

                txt = gr.Textbox(
                    placeholder="Ask something...",
                    show_label=False,
                    scale=8
                )

                send_btn = gr.Button(
                    "Send",
                    scale=1
                )

    # =====================================================
    # CREATE NEW CHAT
    # =====================================================
    new_chat_btn.click(
        create_topic,
        inputs=[sessions_state],
        outputs=[
            sessions_state,
            topic_list,
            chatbot
        ]
    )

    # =====================================================
    # SWITCH CHAT
    # =====================================================
    topic_list.change(
        switch_topic,
        inputs=[
            topic_list,
            sessions_state
        ],
        outputs=[chatbot]
    )

    # =====================================================
    # SEND MESSAGE
    # =====================================================
    send_btn.click(
        chat_stream,
        inputs=[
            txt,
            chatbot,
            topic_list,
            sessions_state
        ],
        outputs=[
            txt,
            chatbot,
            sessions_state
        ]
    )

    txt.submit(
        chat_stream,
        inputs=[
            txt,
            chatbot,
            topic_list,
            sessions_state
        ],
        outputs=[
            txt,
            chatbot,
            sessions_state
        ]
    )


# =========================================================
# 10. RUN
# =========================================================
app.queue()
app.launch(inbrowser=True)
