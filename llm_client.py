from sarvamai import SarvamAI
from secret_key import sarvam_ai_key
import os
import re


class LLMClient:

    def __init__(self):
        os.environ["SARVAM_API_KEY"] = sarvam_ai_key
        self.client = SarvamAI(api_subscription_key=os.environ['SARVAM_API_KEY'])

    # -------- CLASSIFIER -------- #
    def is_farming_question(self, question: str) -> bool:
        """Ask Sarvam AI to classify if the question is farming-related or casual."""
        try:
            response = self.client.chat.completions(
                model="sarvam-m",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a classifier. Respond with ONLY one word: "
                            "'farming' if the message is related to agriculture, crops, "
                            "farming, seeds, soil, pesticides, irrigation, or plants. "
                            "Respond 'casual' for greetings, personal questions, or anything else."
                        )
                    },
                    {"role": "user", "content": question}
                ]
            )
            result = response.choices[0].message.content.lower()
            print(f"🔍 Question type: {result}")
            return result == "farming"

        except Exception as e:
            print(f"❌ Classifier error: {e}")
            return True  # default to farming if classifier fails

    # -------- CASUAL CHAT -------- #
    def casual_chat(self, message: str) -> str:
        """Handle greetings and non-farming questions directly."""
        try:
            response = self.client.chat.completions(
                model="sarvam-m",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are FarmGuru, a friendly assistant for Indian farmers. "
                            "For casual messages like greetings, respond warmly and briefly. "
                            "Always remind the user you are here to help with farming questions."
                        )
                    },
                    {"role": "user", "content": message}
                ]
            )
            return response.choices[0].message.content[7:]

        except Exception as e:
            print(f"❌ Casual chat error: {e}")
            return "Hello! I'm FarmGuru. Ask me anything about farming! 🌱"

    # -------- FARMING ANSWER -------- #
    def ask(self, context: str, question: str) -> str:
        """Answer a farming question using the provided PDF context."""
        prompt = f"""
        Answer the question using ONLY the context below.
        Context:
        {context}
        Question:
        {question}
        Answer clearly:
        """
        try:
            response = self.client.chat.completions(
                model="sarvam-m",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert agricultural assistant helping Indian farmers. "
                            "Answer only from the given context. "
                            "If the answer is not in the context, say: "
                            "'I don't have information about this in the provided documents.'"
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content[7:]
        except Exception as e:
            print(f"❌ Sarvam AI error: {e}")
            return "Sorry, I was unable to get an answer. Please try again."
