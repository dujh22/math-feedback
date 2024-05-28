import os  # Import the os module to interact with the operating system.
from collections import defaultdict  # Import defaultdict from collections to create dictionaries with default values.
from multiprocessing.pool import ThreadPool  # Import ThreadPool to execute parallel tasks.
from typing import Any  # Import Any from typing for type annotations.

import jinja2  # Import the Jinja2 library for templating HTML.
import numpy as np  # Import numpy for numerical operations.
from tqdm import tqdm  # Import tqdm for displaying progress bars.

from .types import EvalResult, Message, SingleEvalResult  # Import custom types from a local module.

def _compute_stat(values: list, stat: str):
    # Function to compute statistical measures (mean, standard deviation, min, max) for a list of values.
    if stat == "mean":
        return np.mean(values)  # Return the mean of values.
    elif stat == "std":
        return np.std(values)  # Return the standard deviation of values.
    elif stat == "min":
        return np.min(values)  # Return the minimum value.
    elif stat == "max":
        return np.max(values)  # Return the maximum value.
    else:
        raise ValueError(f"Unknown {stat =}")  # Raise an error if stat is not recognized.

def aggregate_results(
    single_eval_results: list[SingleEvalResult],
    default_stats: tuple[str] = ("mean", "std"),
    name2stats: dict[str, tuple[str]] | None = None,
) -> EvalResult:
    """
    Aggregate results from multiple evaluations into a single EvalResult.
    """
    name2stats = name2stats or {}  # Use the provided name2stats dictionary or an empty dictionary.
    name2values = defaultdict(list)  # Create a defaultdict to collect values for each metric.
    htmls = []  # List to store HTML snippets.
    convos = []  # List to store conversation excerpts.
    for single_eval_result in single_eval_results:
        for name, value in single_eval_result.metrics.items():
            name2values[name].append(value)  # Append each metric value to the corresponding list.
        if single_eval_result.score is not None:
            name2values["score"].append(single_eval_result.score)  # Append score if it exists.
        htmls.append(single_eval_result.html)  # Collect HTML snippets.
        convos.append(single_eval_result.convo)  # Collect conversations.
    final_metrics = {}  # Dictionary to store final computed metrics.
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)  # Get stats to compute or use default.
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"  # Format key for storage.
            final_metrics[key] = _compute_stat(values, stat)  # Compute and store stat.
    return EvalResult(
        score=final_metrics.pop("score", None), metrics=final_metrics, htmls=htmls, convos=convos
    )  # Return the aggregated results.

def map_with_progress(f: callable, xs: list[Any], num_threads: int = 50):
    """
    Apply function f to each element of list xs using ThreadPool, displaying progress.
    """
    if os.getenv("debug"):
        return list(map(f, tqdm(xs, total=len(xs))))  # Use map and tqdm if in debug mode.
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            return list(tqdm(pool.imap(f, xs), total=len(xs)))  # Use ThreadPool and tqdm otherwise.

jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),  # Set the template loader.
    undefined=jinja2.StrictUndefined,  # Set behavior for undefined variables.
    autoescape=jinja2.select_autoescape(["html", "xml"]),  # Enable auto-escaping for HTML and XML.
)
_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }} 
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""  # HTML template for displaying a message.

def message_to_html(message: Message) -> str:
    """
    Convert a message dictionary to an HTML snippet using the Jinja template.
    """
    return jinja_env.from_string(_message_template).render(
        role=message["role"], content=message["content"], variant=message.get("variant", None)
    )  # Render HTML from the template and message data.

jinja_env.globals["message_to_html"] = message_to_html  # Add message_to_html function to Jinja environment globals.

_report_template = """<!DOCTYPE html>
<html>
    <head>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html }}
    <hr>
    {% endfor %}
    </body>
</html>
"""  # HTML template for generating a report with metrics and examples.

def make_report(eval_result: EvalResult) -> str:
    """
    Generate a standalone HTML report from an EvalResult.
    """
    return jinja_env.from_string(_report_template).render(
        score=eval_result.score,
        metrics=eval_result.metrics,
        htmls=eval_result.htmls,
    )  # Render report HTML from the template and evaluation results.

def make_report_from_example_htmls(htmls: list[str]):
    """
    Generate a standalone HTML report from a list of example HTML snippets.
    """
    return jinja_env.from_string(_report_template).render(score=None, metrics={}, htmls=htmls)  # Render report HTML without score or metrics.
