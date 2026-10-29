import os
from pathlib import Path
import requests
import json
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
import sys
import time
import traceback
import itertools
from nicegui import ui
try:
    import sounddevice as sd
except ImportError:
    print("sounddevice module not found. Audio playback will be disabled.")
Question = ""
Voice_instruction = ("Said in an angry Scottish accent")
resp_instruction = ("You are a helpful assistant who speaks like a scottish person. Give answers under 3 lines. Give a possible google search to find more information labelled as 'Google search'. If you do not know the answer, say 'I do not know' and give a possible google search to find more information labelled as 'Google search'")

Voice_toggle = ""

url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")

project_audio_path = Path(__file__).resolve().parent / "assets" / "Sean_Connery.wav"
downloads_audio_path = Path.home() / "Downloads" / "cry.wav"
soundfile_path = str(
    downloads_audio_path if downloads_audio_path.is_file() else project_audio_path
)

def respond(question, llm_instruction, Voice_toggle):
    print(f"Generating response for question: {question}")

    from pyexpat import model
    import ollama
    from ollama import Client
    from llm_axe import OllamaChat, DataExtractor
    from llm_axe.core import read_website

    llm = OllamaChat(model="gemma4")
    de = DataExtractor(llm)

    client = Client(headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"})
    
    payload = {
        "model": "gemma4", 
        "messages": [
            {'role': 'system', 'content': llm_instruction},
            {"role": "user", "content": question}
        ]
    }
    response_short = requests.post(url, json=payload, stream=True)
    full_reply = ""

    if response_short.status_code == 200:
        print("Streaming response from Ollama:")
        for line in response_short.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        chunk = json_data["message"]["content"]
                        print(chunk, end="")
                        full_reply += chunk
                        globals()['output_label'].set_text(full_reply)

                    if json_data.get("done"):
                        print("\nStreaming complete.")
                        break

                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()
        if Voice_toggle == True:
            print("Starting TTS generation...")
            start = time.time()
            wavs, sr = TTSmodel.generate_voice_clone(
                text=full_reply,
                ref_audio=soundfile_path,
                ref_text="I promise you, when this is over you can cry all you want and I won't say a word.",
                language="English",
            )
            audio_output = wavs[0].astype('float32')
            sd.play(audio_output, sr)
            sd.wait()
            print("Audio playback finished.")
            print(f"Generation took {time.time() - start:.1f}s")
        else:
            print("TTS generation skipped as per user choice.")
    else:
        print("Error: Failed to get a response from Ollama.")
        print(f"Error: {response_short.status_code}")
        print(response_short.text)

    globals()['Full_reply'] = full_reply
    globals()['finished_offline_resp'] = True
    return full_reply


finished_offline_resp = False
Next_steps = False

def submit_question(question_input, Voice_toggle, Next_steps):
    Question = question_input
    globals()['Question'] = Question
    globals()['submitted_question_label'].set_text(f"Submitted question: {Question}")

    print(f"User question: {Question}")
    print("Response generation in progress...")
    respond(question=Question, llm_instruction = globals()['resp_instruction'], Voice_toggle=Voice_toggle)
    if globals()['finished_offline_resp']:
        globals()['output_label'].set_text(globals()['Full_reply'])
        print("Response generation completed.")
        if Next_steps:
            google_search(globals()['Question'])

def google_search(query):
    try:
        from llm_axe import OnlineAgent, OllamaChat

        searcher = OnlineAgent(OllamaChat(model="gemma4"))


        print("Performing Google search")
        if not os.getenv("OLLAMA_API_KEY"):
            print("No API key found. Set the OLLAMA_API_KEY environment variable to perform Google searches.")
            return

        search_question = searcher.get_search_query(query)
        if not search_question:
            print("Could not generate a search query.")
            return

        response_web = searcher.web_search(search_question)
        if response_web:
            print("\nGoogle search results:")
            for idx, result in enumerate(response_web.results, start=1):
                print(f"{idx}. {result.title}: {result.url}")
        else:
            print("No results found.")

    except Exception as e:
        print(f"An error occurred during the Google search: {e}")
        traceback.print_exc()

def main():
    global TTSmodel

    TTSmodel = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="auto",
    )

    sf.info(soundfile_path)

    ui.label("Voice Assistant").style("font-size: 24px; font-weight: bold; color: blue;")
    ui.dark_mode()
    with ui.expansion("Settings"):
        dark = ui.dark_mode()
        ui.label('Switch mode:')
        ui.button('Dark', on_click=dark.enable)
        ui.button('Light', on_click=dark.disable)
        soundfile_path_input = ui.input("Path to reference audio file:", value=soundfile_path)
        soundfile_path_input.on('input', lambda e: globals().update({'soundfile_path': e.value}))
        localhost_input = ui.input("Ollama API URL:", value=url)
        localhost_input.on('input', lambda e: globals().update({'url': e.value}))
    ui.separator()
    with ui.column().classes('justify-center items-center'):
        ui.label("Welcome to the Voice Assistant!").style("font-size: 18px; color: green;")
        with ui.card().classes('fixed_centre'):
            ui.label("Ask a question to the voice assistant:")
            question_input_box = ui.input(placeholder="Type your question here...",).props('autofocus')
            voice_toggle_button = ui.checkbox("Do you want the answer to be spoken?")
            google_search_checkbox = ui.checkbox("Do you want to perform a Google search for more information?")
            ui.button("Submit", on_click=lambda: submit_question(question_input_box.value, voice_toggle_button.value, google_search_checkbox.value))
            submitted_question_label = ui.label("")
        with ui.card().classes('items-center'):
            output = ui.label("")
            globals()['output_label'] = output
            globals()['submitted_question_label'] = submitted_question_label
            ui.label("Voice assistant is running. You can ask questions and get responses.").style("font-size: 16px; color: green;")

    ui.run(
        host="0.0.0.0",
        port=8080,
        reload=False,
        show=False,
        storage_secret=os.environ.get("STORAGE_SECRET", "change-me"),
    )


if __name__ == "__main__":
    main()
