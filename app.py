import streamlit as st
from llm_chains import load_normal_chain,load_pdf_chat_chain
import os
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
import yaml
import json
from streamlit_mic_recorder import mic_recorder
from langchain.schema.messages import HumanMessage, AIMessage
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
# from langchain.memory import StreamlitChatMessagesHistory
from audio_handler import transcribe_audio
from image_handler import handle_image
from pdf_handler import add_documents_to_db


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key
# def save_chat_history_json(chat_history, file_path):
#     with open(file_path, "w") as f:
#         json_data = [message.dict() for message in chat_history]
#         json.dump(json_data, f)

# def save_chat_history():
#     if(st.session_state.history!=[]):
#         if st.session_state.session_key == "new_session":
#             st.session_state.new_session_key = get_timestamp()+".json"
#             save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
#         else:
#             save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key + ".json")
#         # print("Saved and Done")

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            timestamp = get_timestamp()
            st.session_state.new_session_key = timestamp
            filename = f"{timestamp}.json"
            save_chat_history_json(st.session_state.history, os.path.join(config["chat_history_path"], filename))
        else:
            save_chat_history_json(st.session_state.history, os.path.join(config["chat_history_path"], st.session_state.session_key))


def main():
    st.title("Multimodal Chat App")
    chat_container = st.container()
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])

    if "session_key" not in st.session_state:
        st.session_state.session_key = "new_session"

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key=None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key !=None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    
    st.sidebar.selectbox("Select a chat Session", chat_sessions, key="session_key", index=index, on_change=track_index)
    st.toggle("PDF Chat", key="pdf_chat", value=False)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history=[]

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history = chat_history)  # Pass chat_history to load_chain
    llm_chain = load_chain(chat_history)  # Pass chat_history to load_chain

    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)
    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True) 
    with send_button_column:
        send_button = st.button("Send", key="send_button", on_click=clear_input_field)

    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["png", "jpeg", "jpg"])
    uploaded_pdf = st.sidebar.file_uploader("Upload an pdf file", accept_multiple_files=True,key="pdf_upload" ,type=["pdf"], on_change=toggle_pdf_chat)



    print(voice_recording)

    if uploaded_pdf:
        with st.spinner("Processing pdfs....."):
            add_documents_to_db(uploaded_pdf)

    if uploaded_audio :
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        llm_chain.run("Summarize this text: "+transcribed_audio, chat_history)

    if(voice_recording):
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        llm_chain.run(transcribed_audio, chat_history)
        print(transcribed_audio)

    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing Image....."):
                user_message = "Describe this image in detail please."
                if st.session_state.user_question !=[]:
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_answer = handle_image(uploaded_image.getvalue(), st.session_state.user_question)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)

        if st.session_state.user_question != "":
            # st.chat_message("user").write(st.session_state.user_question)
            llm_response = llm_chain.run(st.session_state.user_question, chat_history)  # Pass chat_history here
            print(llm_response)
            # st.chat_message("ai").write(llm_response)
            st.session_state.user_question = ""
    if chat_history.messages !=[]:
        with chat_container:
            st.write("Chat History: ")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)
    # print(chat_history.messages[0].dict())
    save_chat_history()
    print(chat_sessions)
# print("test")
 

if __name__ == "__main__":
    main()


