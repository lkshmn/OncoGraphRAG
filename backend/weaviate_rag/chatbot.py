import chainlit as cl
from chainlit.input_widget import Select, Slider
from openai import OpenAI
from weaviate_rag.rag_system import GraphRAGSystem
import time
import logging
import re
import sys
import os

# Ensure backend modules are reachable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Assuming prompts.py is in backend/prompts.py
from backend.prompts import generate_chat_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LM Studio Configuration
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio" # Local server usually ignores this, but client requires it
# Default model identifier
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model") 

MAX_HISTORY = 10
TEMPERATURE = 0.3

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)


def sanitize_input(text):
    """Enhanced sanitization to prevent prompt injection and control character attacks."""
    if text is None:
        return None

    text = text.replace('\r', '').replace('\x08', '')

    if len(text) > 2000: # Increased limit for medical queries
        logger.warning(f"Input exceeds maximum length: {len(text)} chars")
        return None

    injection_patterns = [
        r'(?i)\bignore\b.*\binstructions\b',
        r'(?i)\bdisregard\b.*\binstructions\b',
        r'(?i)\bforget\b.*\bguidelines\b',
        r'(?i)\bjust\s+say\b',
        r'(?i)\bjust\s+print\b',
        r'(?i)\boutput\s+the\s+following\b',
        r'(?i)\bverbatim\b',
        r'(?i)\bSTAN\b',
        r'(?i)\bDAN\b',
        r'(?i)\bDUDE\b',
        r'(?i)\bANTI-DAN\b'
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text):
            logger.warning(f"Potential prompt injection detected: {text[:50]}...")
            return None

    return text


def retrieve(user_input):
    try:
        sanitized_input = sanitize_input(user_input)

        if sanitized_input is None:
            return "No relevant information available due to potential security concerns."

        # Initialize RAG System
        analyzer = GraphRAGSystem(sanitized_input)
        context = analyzer.analyze()

        return context
    except Exception as e:
        logger.exception("Error in retrieve function")
        return f"Error retrieving information: {str(e)}"


async def create_system_prompt(user_input):
    sanitized_input = sanitize_input(user_input)

    context = retrieve(sanitized_input)
    logger.info("Retrieved context for user query")
    # Using the shared prompt generator
    system_prompt = generate_chat_prompt(context, sanitized_input)

    return system_prompt


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("history", [])
    cl.user_session.set("model", LMSTUDIO_MODEL)
    cl.user_session.set("temperature", TEMPERATURE)

    # Simplified settings for local usage
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Select Model",
                values=[LMSTUDIO_MODEL],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=TEMPERATURE,
                min=0.1,
                max=1.0,
                step=0.1,
            )
        ]
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    if "model" in settings:
        cl.user_session.set("model", settings["model"])
    if "temperature" in settings:
        cl.user_session.set("temperature", float(settings["temperature"]))


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Lung Cancer Symptoms",
            message="What are the common early symptoms of Non-Small Cell Lung Cancer?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Immunotherapy Research",
            message="What are the latest clinical trial findings for immunotherapy in Triple-Negative Breast Cancer?",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Treatment Comparison",
            message="Compare the efficacy of chemotherapy vs targeted therapy for advanced Melanoma.",
            icon="/public/write.svg",
        )
    ]


@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    if user_input.lower() == "/clear":
        cl.user_session.set("history", [])
        await cl.Message("🔄 Chat history cleared. Starting fresh!").send()
        return

    sanitized_input = sanitize_input(user_input)

    if sanitized_input is None:
        await cl.Message(
            content="I can only provide factual information about oncology research. Your question contains patterns that may compromise security or exceed length limits. Please rephrase your question.").send()
        return
    
    # Generate system prompt with RAG context
    system_prompt = await create_system_prompt(sanitized_input)

    history = cl.user_session.get("history", [])

    # Ensure system prompt is fresh for every turn to include new retrieved context
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": system_prompt})
    else:
        history[0]["content"] = system_prompt

    history.append({"role": "user", "content": sanitized_input})

    # Keep history within limits
    if len(history) > MAX_HISTORY * 2 + 1:
        history = [history[0]] + history[-(MAX_HISTORY * 2):]

    model_name = cl.user_session.get("model", LMSTUDIO_MODEL)
    temperature = cl.user_session.get("temperature", TEMPERATURE)

    try:
        msg = cl.Message(content="")
        await msg.send()
        
        # Stream response from Local LLM
        stream = client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", history)

    except Exception as e:
        logger.exception("Error in chat processing")
        error_msg = f"An error occurred: {str(e)}"
        await cl.Message(content=error_msg).send()
