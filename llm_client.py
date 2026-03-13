from sarvamai import SarvamAI
from secret_key import sarvam_ai_key
import os


class LLMClient:

    def __init__(self):
        os.environ["SARVAM_API_KEY"] = sarvam_ai_key
        self.client = SarvamAI(api_subscription_key=os.environ['SARVAM_API_KEY'])

    def ask(self, context: str, question: str) -> str:

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
            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Sarvam AI error: {e}")
            return "Sorry, I was unable to get an answer. Please try again."