from pyexpat import model
import ollama
from ollama import Client
import os
import requests
import json
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
import sounddevice as sd
import sys
import time
import traceback
from llm_axe import OnlineAgent, OllamaChat, PdfReader, DataExtractor
from llm_axe.core import read_website
import urllib.parse
import itertools
from nicegui import ui

client = Client(headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"})

def loading_animation():
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    
    for _ in range(50):
        sys.stdout.write(f"\rLoading your data... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
        
    sys.stdout.write("\rLoading complete!      \n")

def show_loading_animation():
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    for frame in spinner:
        if not globals().get('still_loading', True):
            break
        terminal_message = f"Generating response... {frame}"
        sys.stdout.write(f"\r{terminal_message}")
        sys.stdout.flush()
        globals()['output_label'].set_text(terminal_message)
        ui.update()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

Question = ""
Voice_instruction = ("Said in an angry Scottish accent")
resp_instruction = ("You are a helpful assistant who speaks like a scottish person. Give answers under 3 lines. Give a possible google search to find more information labelled as 'Google search'. If you do not know the answer, say 'I do not know' and give a possible google search to find more information labelled as 'Google search'")

Voice_toggle = ""

sr = 16000 

TTSmodel = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="auto",
)

url = "http://localhost:11434/api/chat"

llm = OllamaChat(model="gemma4")
searcher = OnlineAgent(llm)
de = DataExtractor(llm)

data, sr = sf.read("C:\\Users\\carol\\Downloads\\cry.wav", always_2d=False)
sf.write("C:\\Users\\carol\\Downloads\\cry_fixed.wav", data, sr)

def respond(question, llm_instruction, Voice_toggle):
    print(f"Generating response for question: {question}")
    
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
                ref_audio="C:\\Users\\carol\\Downloads\\cry_fixed.wav",
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
    print("Performing Google search")
    if not os.getenv("OLLAMA_API_KEY"):
        print("No API key found.")
        return

    search_question = searcher.get_search_query(query)
    if not search_question:
        print("Could not generate a search query.")
        return

    response_web = client.web_search(search_question)
    if response_web:
        print("\nGoogle search results:")
        for idx, result in enumerate(response_web.results, start=1):
            print(f"{idx}. {result.title}: {result.url}")
    else:
        print("No results found.")

ui.label("Ask a question to the voice assistant:")
question_input_box = ui.input(placeholder="Type your question here...",).props('autofocus')
voice_toggle_button = ui.checkbox("Do you want the answer to be spoken?")
google_search_checkbox = ui.checkbox("Do you want to perform a Google search for more information?")
submit_button = ui.button("Submit", on_click=lambda: submit_question(question_input_box.value, voice_toggle_button.value, google_search_checkbox.value))
submitted_question_label = ui.label("")
output = ui.label("")
globals()['output_label'] = output
globals()['submitted_question_label'] = submitted_question_label
ui.label("Voice assistant is running. You can ask questions and get responses.").style("font-size: 16px; color: green;")
ui.run()
