from sarvamai import SarvamAI
from secret_key import sarvam_ai_key
import os


class LLMClient:

    def __init__(self, api_key):

        os.environ["SARVAM_API_KEY"] = sarvam_ai_key
        self.client = SarvamAI(api_subscription_key=os.environ['SARVAM_API_KEY'])

    def ask(self, context, question):

        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}

        Answer clearly:
        """

        response = self.client.chat.completions(
            model="sarvam-m",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content