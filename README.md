# DevGuide Assistant

DevGuide Assistant is a task-oriented chatbot for game-development
project guidance and task estimation. It uses Python, FastAPI, and
the OpenAI Responses API.

## Features

- Streamed responses through a browser interface.
- Short-term conversation memory, with an eight-turn default window.
- Model-initiated project knowledge base search and task estimation.
- Validated tool arguments and structured JSON tool results.
- Server-side safety checks, secret redaction, and rate limiting.
- Per-turn latency, token usage, estimated cost, and tool-call logging.

## Requirements

- Python 3.10 or newer.
- An OpenAI API key with available credits and access to the configured model.
- Internet access for live model requests.

## Installation

Open a terminal in the DevGuideAssistant folder.

Create and activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r requirements.txt

Copy the example configuration:

    Copy-Item .env.example .env

Edit the private .env file to provide your API key and configuration.
Do not submit or publish .env.

The evaluated model was GPT-5.4-mini. The evaluation used configured
rates of $0.75 per million input tokens and $4.50 per million output
tokens. Match the configured prices to the selected model when
running a different experiment.

## Run the Web Application

From the project folder:

    python app.py

Open:

- Chatbot: http://localhost:8000
- API documentation: http://localhost:8000/docs

Stop the server with Ctrl+C. Conversation memory resets when the
server restarts.

## API Endpoints

- POST /chat: streamed response using Server-Sent Events.
- POST /chat/json: completed JSON response used by evaluation.
- POST /chat/stream: compatibility alias for the streaming endpoint.
- GET /health: application configuration health check.
- DELETE /conversations/{conversation_id}: clear conversation memory.

Both chat endpoints accept:

    {
        "conversation_id": "demo",
        "user_message": "What prefix should an Actor Component Blueprint use?"
    }

The streaming endpoint emits start, text_delta, tool_call, complete,
and error events. A complete event confirms successful completion;
HTTP 200 alone does not confirm a successful streamed answer.

## Run the Command-Line Interface

    python chat_cli.py

## Run Offline Tests

    python -m pytest -q --tb=short

The test suite uses simulated provider responses rather than paid
model requests. Provide a nonempty placeholder API key in the local
.env file if running the tests without a real key. The verified suite
passed 137 tests before the launch entrypoint was added.

## Run Evaluation

Preserve existing results before running evaluation, because the
scripts overwrite evaluation files.

    python -m evaluation.evaluate

This makes live API requests for a 20-prompt baseline and two
five-prompt memory-window conditions. API usage incurs charges.

Review the saved responses in results/manual_review.csv. Enter 0 or 1
for manual task success and hallucination, and document the reasoning.
Score the saved first-attempt responses; record retries separately.

Then run:

    python -m evaluation.score_results
    python -m evaluation.generate_charts

## Verified Evaluation Results

The baseline used 20 prompts covering tools, memory, safety, and
edge cases.

- Automatic task success: 95% (19/20).
- Manual task success: 100% (20/20).
- Observed hallucinations in reviewed cases: 0.
- Average request latency: 1,539.453 ms.
- Total recorded tokens: 16,831.
- Estimated baseline API cost: $0.01637325.
- Successful tool calls: 10 out of 10.

E01 passed manual review but failed automatic phrase matching.
Both memory-window conditions achieved 100% automatic task success
on the five-prompt sequence. These results apply to the tested
scenarios and do not establish general reliability.

## Output Locations

- ***results/metrics.json*:** baseline evaluation summary.
- ***results/evaluation_results.json*:** baseline responses and scores.
- ***results/manual_review.csv*:** manual scoring worksheet.
- ***results/manual_scoring_summary.json*:** manual score summary.
- ***results/transcripts/*:** individual baseline prompt records.
- ***results/runtime_metrics/*:** baseline runtime logs.
- ***results/ablation_results.json*:** memory-window comparison.
- ***results/ablation/*:** condition-specific transcripts and runtime logs.
- ***results/figures/*:** generated charts.
- ***results/interactive_runtime/*:** browser and CLI logs when configured
  through ***RESULTS_DIRECTORY***.

## Limitations

Conversation memory is process-local and retains a rolling window
of complete turns. Persistent vector-based long-term memory is not
implemented. Saved transcripts are evaluation records, not retrieved
conversation memory.

Safety checks cover selected extraction and instruction-override
patterns. They are not comprehensive moderation. The application is
a local coursework demonstration and does not implement authenticated
conversation ownership or a production deployment security model.

Latency measures completed JSON requests rather than time to first
streamed text. Estimated costs use configured rates rather than an
independently reconciled billing total.

## Submission Security

Exclude .env, Git history, virtual environments, IDE settings, Python
bytecode, caches, and old result backups from the submission archive.
Include .env.example with a placeholder API key.

Localhost addresses work only on the computer running the server.
They are not public chatbot build links.