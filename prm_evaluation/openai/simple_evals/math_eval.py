# Import necessary libraries
import random
import re
import blobfile as bf
import pandas

# Import modules from the current package
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

# Define a template for posing math problem queries
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form ANSWER: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "ANSWER:", and you do not need to use a \\boxed command.
""".strip()

# Regex pattern to find the answer line in the response
ANSWER_PATTERN = r"(?i)ANSWER\s*:\s*([^\n]+)"

# Template to judge the equivalence of two math expressions
EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

[Example scenarios]

---

YOUR TASK

Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

# Function to check if two expressions are equivalent based on the given sampler's response
def check_equality(sampler: SamplerBase, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = sampler([dict(content=prompt, role="user")])
    return response.lower().strip() == "yes"

# Define a class for evaluating mathematical problems
class MathEval(Eval):
    def __init__(self, equality_checker: SamplerBase, num_examples: int | None = None):
        # Load example problems from a CSV file hosted on a public blob
        df = pandas.read_csv(bf.BlobFile("https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"))
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker

    # Method to execute the evaluation
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # Generate a prompt message for each math problem
            prompt_messages = [dict(content=QUERY_TEMPLATE.format(**row), role="user")]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        # Aggregate and return results of all evaluations
        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)

# HTML template for displaying results in a web-based interface
HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }}</p>
<p>Extracted Answer: {{ extracted_answer }}</p>
<p>Score: {{ score }}</p>
"""
