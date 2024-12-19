from langchain_openai import ChatOpenAI
import dspy
from typing import List, Dict
from dataclasses import dataclass


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
            f"{'Channel 1' if item.get('channel') == 1 else 'Channel 0'}: {item['phrase'].strip()}"
            for item in transcription["items"]
        )

        prompt = f"""Analyze this conversation and determine if each statement is true ('Так') or false ('Ні'):

Conversation:
{conversation}"""

        res = structured_llm.invoke(prompt)
        return res, "No reasoning"


@dataclass
class Utterance:
    channel: int
    phrase: str
    timestamp: float


@dataclass
class Question:
    text: str
    possible_answers: List[str]


class DialogueAudit(dspy.Signature):
    """Audit dialogue based on provided questions."""

    dialogue: List[Utterance] = dspy.InputField(desc="List of dialogue utterances")
    questions: List[Question] = dspy.InputField(
        desc="List of questions to evaluate the dialogue"
    )
    answers: List[str] = dspy.OutputField(
        desc="List of answers corresponding to questions"
    )


class DSPY_auditor:
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """Initialize the DSPY auditor with API key and model name."""
        self.lm = dspy.LM("openai/" + model_name, api_key=api_key, cache=False)
        dspy.configure(lm=self.lm)
        # self.predictor = dspy.Predictor(DialogueAudit)
        self.predictor = dspy.ChainOfThought(DialogueAudit)

    def audit(
        self, transcription: Dict[str, List[Dict]], questions_with_options: List[tuple]
    ) -> Dict[str, str]:
        """
        Audit a conversation transcript using the provided questions.

        Args:
            transcription: Dictionary containing conversation items
            questions_with_options: List of tuples containing (question, possible_answers)

        Returns:
            Dictionary mapping questions to their answers
        """
        # Convert transcription to list of Utterances
        dialogue = [
            Utterance(
                channel=item.get("channel"),
                phrase=item["phrase"].strip(),
                timestamp=item.get("start_sec", 0),
            )
            for item in transcription["items"]
        ]

        # Convert questions to DSPY format
        questions_data = [
            Question(text=question, possible_answers=[opt for opt in options])
            for question, options in questions_with_options
        ]

        # Get predictions using DSPY
        result = self.predictor(dialogue=dialogue, questions=questions_data)
        reasoning = result.reasoning
        # Format output to match LLM_auditor
        return {
            question: answer
            for question, answer in zip(
                [q for q, _ in questions_with_options], result.answers
            )
        }, reasoning
