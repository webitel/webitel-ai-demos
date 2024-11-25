from langchain_openai import ChatOpenAI


class LLM_auditor:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=api_key)

    def audit(self, transcription, questions_with_options):
        json_schema = {
            "title": "audit_response",
            "description": "Audit responses for conversation analysis",
            "type": "object",
            "properties": {
                question: {
                    "type": "string",
                    "description": f"Answer for: {question}",
                    "enum": options,
                }
                for question, options in questions_with_options
            },
            "required": [question for question, options in questions_with_options],
        }

        structured_llm = self.llm.with_structured_output(json_schema)

        conversation = "\n".join(
            f"{'Customer' if item.get('channel') == 1 else 'Operator'}: {item['phrase'].strip()}"
            for item in transcription["items"]
        )

        prompt = f"""Analyze this conversation and determine if each statement is true ('Так') or false ('Ні'):

Conversation:
{conversation}"""

        res = structured_llm.invoke(prompt)
        return res
