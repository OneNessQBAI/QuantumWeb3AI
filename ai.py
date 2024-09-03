import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import openai
import json
from web3 import Web3
import argparse
import pygame
import speech_recognition as sr
from gtts import gTTS
from googleapiclient.discovery import build
import requests
import sys
from PIL import Image
import io
import base64
import cirq
import numpy as np

# Initialize Pygame mixer
pygame.mixer.init()

# Load your configuration
config = {
    "rpc_url1": "https://bsc-dataseed.bnbchain.org/",
    "rpc_url2": "https://bsc-dataseed-public.bnbchain.org",
    "private_key": "METAMASK_PRIVATE_KEY",
    "chain_id": 56,
    "openai_key": "YOUR _OPENAI_API_KEY"
}

# Connect to the blockchain
web3 = Web3(Web3.HTTPProvider(config['rpc_url1']))

# Your contract ABI (replace with the actual ABI)
contract_abi = [
    {
        "constant": False,
        "inputs": [
            {
                "name": "_question",
                "type": "string"
            }
        ],
        "name": "sendMessage",
        "outputs": [],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

class QuantumDecisionMaker:
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.qubits = cirq.LineQubit.range(num_qubits)
        self.circuit = cirq.Circuit()

    def initialize_superposition(self):
        self.circuit.append(cirq.H.on_each(*self.qubits))

    def apply_oracle(self, transaction_data):
        for i, qubit in enumerate(self.qubits):
            if i < len(transaction_data) and transaction_data[i] == '1':
                self.circuit.append(cirq.X(qubit))
        self.circuit.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
        for i, qubit in enumerate(self.qubits):
            if i < len(transaction_data) and transaction_data[i] == '1':
                self.circuit.append(cirq.X(qubit))

    def apply_diffusion(self):
        self.circuit.append(cirq.H.on_each(*self.qubits))
        self.circuit.append(cirq.X.on_each(*self.qubits))
        self.circuit.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
        self.circuit.append(cirq.X.on_each(*self.qubits))
        self.circuit.append(cirq.H.on_each(*self.qubits))

    def run_grover(self, transaction_data, iterations):
        self.initialize_superposition()
        for _ in range(iterations):
            self.apply_oracle(transaction_data)
            self.apply_diffusion()
        return self.circuit

    def measure(self):
        self.circuit.append(cirq.measure(*self.qubits, key='result'))
        simulator = cirq.Simulator()
        result = simulator.run(self.circuit, repetitions=100)
        counts = result.histogram(key='result')
        return max(counts, key=counts.get)

def quantum_validate_transaction(transaction):
    transaction_data = ''.join(format(ord(c), '08b') for c in json.dumps(transaction))
    qdm = QuantumDecisionMaker()
    iterations = int(np.pi/4 * np.sqrt(2**8))
    qdm.run_grover(transaction_data, iterations)
    result = qdm.measure()
    return result % 2 == 0

def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def capture_audio(timeout=50, phrase_time_limit=50):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def get_openai_response(prompt, system_content, image_path=None):
    client = openai.OpenAI(api_key=config['openai_key'])
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    if image_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
            ]
        })
    
    response = client.chat.completions.create(
        model="gpt-4o" if image_path else "gpt-4-1106-preview",
        messages=messages,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def generate_and_play_tts(text):
    tts = gTTS(text=text, lang='en')
    audio_file_path = "response.mp3"
    tts.save(audio_file_path)
    play_audio(audio_file_path)

def send_transaction_rpc(to_address, value):
    account = web3.eth.account.from_key(config['private_key'])
    nonce = web3.eth.get_transaction_count(account.address)
    gas_estimate = web3.eth.estimate_gas({
        'to': to_address,
        'from': account.address,
        'value': web3.to_wei(float(value.replace(' BNB', '').replace(' ETH', '')), 'ether')
    })
    gas_price = web3.eth.gas_price
    tx = {
        'nonce': nonce,
        'to': to_address,
        'value': web3.to_wei(float(value.replace(' BNB', '').replace(' ETH', '')), 'ether'),
        'gas': gas_estimate,
        'gasPrice': gas_price,
        'chainId': config['chain_id']
    }
    signed_tx = web3.eth.account.sign_transaction(tx, config['private_key'])
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return web3.to_hex(tx_hash)

def send_message(to_address, input_data):
    contract = web3.eth.contract(address=to_address, abi=contract_abi)
    account = web3.eth.account.from_key(config['private_key'])
    nonce = web3.eth.get_transaction_count(account.address)
    gas_estimate = contract.functions.sendMessage(input_data['string _question']).estimateGas({
        'from': account.address
    })
    gas_price = web3.eth.gas_price
    tx = contract.functions.sendMessage(input_data['string _question']).build_transaction({
        'nonce': nonce,
        'gas': gas_estimate,
        'gasPrice': gas_price,
        'chainId': config['chain_id']
    })
    signed_tx = web3.eth.account.sign_transaction(tx, config['private_key'])
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    return web3.to_hex(tx_hash)

def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res.get('items', [])

def process_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((224, 224))
        return img

def generate_image(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai_key']}"
    }
    data = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, data=json.dumps(data))
    return response.json()

def process_transaction(transaction_data):
    to_address = transaction_data.get("to_address")
    value = transaction_data.get("value")
    is_valid = quantum_validate_transaction(transaction_data)
    if is_valid:
        tx_hash = send_transaction_rpc(to_address, value)
        return f"Transaction validated and sent. Hash: {tx_hash}"
    else:
        return "Transaction failed quantum validation."

def main():
    parser = argparse.ArgumentParser(description='Process a prompt for OpenAI.')
    parser.add_argument('--prompt', type=str, help='The prompt for OpenAI')
    parser.add_argument('--audio', action='store_true', help='Use audio input')
    parser.add_argument('--text', action='store_true', help='Print response as text')
    parser.add_argument('--google', type=str, help='Perform a Google search and return results')
    parser.add_argument('--image', type=str, help='Path to an image file for vision analysis')
    parser.add_argument('--generate-image', type=str, help='Generate an image based on the prompt')
    args = parser.parse_args()

    if args.google:
        try:
            api_key = 'AIzaSyC6-sY2-MBDJDMYrPBIvLnzVxq27sNm6j4'
            cse_id = '62b0fb1f3896441e3'
            results = google_search(args.google, api_key, cse_id)
            if results:
                for result in results:
                    print(result['title'], result['link'])
            else:
                print("No results found.")
        except Exception as e:
            print(f"Error performing Google search: {e}")
        return

    if args.audio:
        prompt = capture_audio(timeout=120, phrase_time_limit=120)
        if not prompt:
            print("Failed to capture audio or recognize speech.")
            return
    else:
        prompt = args.prompt

    if args.generate_image:
        response = generate_image(args.generate_image)
        print(json.dumps(response, indent=2))
        return

    if not prompt and not args.image:
        print("No prompt or image provided.")
        return

    system_content = """
       You provide data on Quantum and blockchain technology 
    """

    if args.image:
        image = process_image(args.image)
        response = get_openai_response(prompt or "Describe this image", system_content, args.image)
    else:
        response = get_openai_response(prompt, system_content)

    print(response)

    try:
        response_data = json.loads(response)
        action = response_data.get("action")
        if action == "sendTransaction":
            result = process_transaction(response_data)
            print(result)
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")

if __name__ == "__main__":
    main()
