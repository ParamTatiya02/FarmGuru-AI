from secret_key import sarvam_ai_key
import os
import streamlit as st
from sarvamai import SarvamAI

#Initialize the Sarvam AI secret API key
os.environ['SARVAM_API_KEY'] = sarvam_ai_key
client = SarvamAI(api_subscription_key=os.environ['SARVAM_API_KEY'])

#Web page
# st.title("FarmGuru Chatbot")

#Response
# Send a message to the API
response = client.chat.completions(
    model="sarvam-m",
    messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message.content)