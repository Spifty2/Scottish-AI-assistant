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

client = Client(headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"})

def loading_animation():
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    
    for _ in range(50):
        sys.stdout.write(f"\rLoading your data... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
        
    sys.stdout.write("\rLoading complete!      \n")

Question = input("Ask a question:  ")
Voice_instruction = ("Said in an angry Scottish accent")

Voice_toggle = input("Do you want the answer to be spoken? (yes/no):  ")

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

def respond(question, llm_instruction):
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

                    if json_data.get("done"):
                        print("\nStreaming complete.")
                        break

                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()
        if Voice_toggle.lower() == "yes":
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

    return full_reply

full_reply = respond(
    Question,
    ' You are a helpful assistant who speaks like a scottish person. Give answers under 3 lines. Give a possible google search to find more information labelled as "Google search". If you do not know the answer, say "I do not know" and give a possible google search to find more information labelled as "Google search"'
)

print("Offline response complete")

Next_steps = input("Do you want to search Google for more information? (yes/no):  ")

def google_search(query):
    search_question = de.ask(query, "find the best possible google search to find more information about this question")
    response_web = client.web_search(search_question)
    if response_web:
        print("\nGoogle search results:")
        for idx, result in enumerate(response_web.results, start=1):
            print(f"{idx}. {result.title}: {result.url}")
    else:
        print("No results found.")
    

if Next_steps.lower() == "yes":
    print("Performing Google search...")
    start_time = time.time()
    google_results_text = google_search(full_reply)
    end_time = time.time()
    print(f"Google search completed in {end_time - start_time:.2f} seconds.")
else:
    print("No further Google search will be performed.")

