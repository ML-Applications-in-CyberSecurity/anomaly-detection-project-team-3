import socket
import json
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
import requests


HOST = 'localhost'
PORT = 9999

model = joblib.load("D:/project/anomaly_model.joblib")

def pre_process_data(data):
    # Convert data to DataFrame for model prediction
    df = pd.DataFrame([data])
    #TODO 2: Here you have to add code to pre-process the data as per your model requirements.
    df = pd.get_dummies(df, columns=['protocol'], dtype=int)
    return df.to_numpy()


def call_together_ai_llm(prompt):
    url = "https://api.together.ai/v1/chat/completions"
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-70b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are an expert anomaly detection alert writer."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        return f"Error from LLM API: {response.status_code} {response.text}"


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                #TODO 3: Here you have to add code to process the received data and detect anomalies using a trained model.
                processed_data = pre_process_data(data)
                prediction = model.predict(processed_data)[0]  # 1=normal, -1=anomaly

                load_dotenv()  # Load environment variables from .env

                api_key = os.getenv("TOGETHER_API_KEY")

                # Prepare message for LLM
                alert_type = "Anomaly" if prediction == -1 else "Normal"

                prompt = (
                    f"Generate a concise alert message for the following data:\n\n"
                    f"{json.dumps(data, indent=2)}\n\n"
                    f"Label: {alert_type}\n\n"
                    "Format the message like this:\n"
                    "'<Label>: <Short reason based on feature values>'\n\n"
                    "Be clear and human-readable, using relevant feature names to justify the label."
                )


                # Call Together AI LLaMA3 70B LLM to generate alert caption
                alert_caption = call_together_ai_llm(prompt)

                print(f"Alert Caption:\n{alert_caption}\n")

            except json.JSONDecodeError:
                print("Error decoding JSON.")
