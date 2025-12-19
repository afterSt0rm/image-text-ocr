import json
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="OCR", layout="wide")

# --- Constants ---
DOCUMENT_TYPES = {
    "Auto Detect": "auto",
    "Image": "image",
    "PDF": "pdf",
    "National ID": "national_id",
    "Offer Letter": "offer_letter",
}

# --- State Management ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "doc_type" not in st.session_state:
    st.session_state.doc_type = "Auto Detect"


def create_new_chat(uploaded_file):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "image": uploaded_file,
        "messages": [],
        "doc_type": st.session_state.doc_type,
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_base_url = st.text_input("API Base URL", value="http://localhost:8000")

    # Document Type Selector
    st.session_state.doc_type = st.selectbox(
        "Document Type",
        options=list(DOCUMENT_TYPES.keys()),
        help="Select the type of document for specialized processing",
    )

    if st.button("Check Health"):
        try:
            response = requests.get(f"{api_base_url}/healthcheck")
            if response.status_code == 200:
                st.success("Backend is Healthy!")
            else:
                st.error(f"Backend returned {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.header("Chat History")

    chat_ids = list(st.session_state.chats.keys())
    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        if st.button(chat_data["title"], key=chat_id, width="stretch"):
            st.session_state.current_chat_id = chat_id

# --- Main Interface ---
st.title("Image Text OCR")

# Image Uploader
uploaded_file = st.file_uploader(
    "Start New Chat with Image or PDF", type=["png", "jpg", "jpeg", "pdf"]
)

if uploaded_file:
    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state.last_uploaded_file != uploaded_file
    ):
        create_new_chat(uploaded_file)
        st.session_state.last_uploaded_file = uploaded_file
        st.rerun()

# Display Current Chat
if (
    st.session_state.current_chat_id
    and st.session_state.current_chat_id in st.session_state.chats
):
    current_chat = st.session_state.chats[st.session_state.current_chat_id]

    # Display Image/PDF for context
    if current_chat["image"]:
        if current_chat["image"].type == "application/pdf":
            st.info(f"PDF Uploaded: {current_chat['image'].name}")
        else:
            st.image(current_chat["image"], caption="Context Image", width=400)

    # Display Message History
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("is_json"):
                st.json(msg["content"])
            else:
                st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about the image or document..."):
        current_chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    current_chat["image"].seek(0)

                    files = {
                        "file": (
                            current_chat["image"].name,
                            current_chat["image"].getvalue(),
                            current_chat["image"].type,
                        )
                    }

                    # Determine endpoint based on document type
                    doc_type = current_chat.get("doc_type", "Auto Detect")
                    doc_type_key = DOCUMENT_TYPES.get(doc_type, "auto")

                    if doc_type_key == "national_id":
                        endpoint = f"{api_base_url}/ocr/national_id"
                        data = {"prompt": prompt}
                    elif doc_type_key == "pdf":
                        endpoint = f"{api_base_url}/ocr/pdf"
                        data = {"prompt": prompt}
                    elif doc_type_key == "image":
                        endpoint = f"{api_base_url}/ocr/image"
                        data = {"prompt": prompt}
                    elif doc_type_key == "offer_letter":
                        endpoint = f"{api_base_url}/ocr/offer_letter"
                        data = {"prompt": prompt}
                    else:  # Auto detect
                        if current_chat["image"].type == "application/pdf":
                            endpoint = f"{api_base_url}/ocr/pdf"
                        else:
                            endpoint = f"{api_base_url}/ocr/image"
                        data = {"prompt": prompt}

                    response = requests.post(endpoint, files=files, data=data)

                    if response.status_code == 200:
                        result = response.json()

                        # Handle different response formats
                        if doc_type_key in ["national_id", "offer_letter"]:
                            # Structured output - display as JSON
                            response_data = result.get("data", {})
                            st.json(response_data)
                            current_chat["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": response_data,
                                    "is_json": True,
                                }
                            )
                        else:
                            response_text = result.get("response", "No response text.")
                            st.markdown(response_text)
                            current_chat["messages"].append(
                                {"role": "assistant", "content": response_text}
                            )
                    else:
                        error_msg = f"Error {response.status_code}: {response.text}"
                        st.error(error_msg)
                        current_chat["messages"].append(
                            {"role": "assistant", "content": error_msg}
                        )

                except Exception as e:
                    error_msg = f"Failed to connect: {e}"
                    st.error(error_msg)
                    current_chat["messages"].append(
                        {"role": "assistant", "content": error_msg}
                    )

elif not uploaded_file:
    st.info("Upload an image to start a conversation.")
