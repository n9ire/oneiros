"""
AI Manager — OpenAI-powered assistant for Oneiros.

The assistant can:
  - Answer ML questions in the context of the user's current graph
  - Suggest architectures for a given dataset
  - Explain validation errors
  - Generate a whole graph from a natural-language prompt

Graph actions are returned inline as a fenced ```actions JSON block
that the frontend parses and applies to the canvas.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are the AI assistant embedded in Oneiros, a visual machine learning IDE.
Users build neural networks by connecting nodes on a canvas.

## Available node types

| type        | parameters (in data object)                                      |
|-------------|------------------------------------------------------------------|
| inputNode   | channels (int), height (int), width (int)                        |
| denseNode   | units (int), activation: relu/sigmoid/tanh/softmax/none          |
| outputNode  | classes (int)                                                    |

## Generating or modifying the graph

When you want to create or change nodes, include EXACTLY ONE fenced ```actions
block in your reply (no other fenced blocks). It must be valid JSON array:

```actions
[
  {"type":"clear"},
  {"type":"addNode","id":"input-1","nodeType":"inputNode","x":80,"y":180,"data":{"channels":1,"height":28,"width":28}},
  {"type":"addNode","id":"dense-1","nodeType":"denseNode","x":330,"y":180,"data":{"units":256,"activation":"relu"}},
  {"type":"addNode","id":"dense-2","nodeType":"denseNode","x":580,"y":180,"data":{"units":128,"activation":"relu"}},
  {"type":"addNode","id":"out-1","nodeType":"outputNode","x":830,"y":180,"data":{"classes":10}},
  {"type":"connect","source":"input-1","target":"dense-1"},
  {"type":"connect","source":"dense-1","target":"dense-2"},
  {"type":"connect","source":"dense-2","target":"out-1"}
]
```

Omit the actions block entirely if you are only explaining or answering a question.

## Rules
- Keep answers concise (3–5 sentences unless asked for more).
- For tabular CSV datasets: set inputNode channels = feature count, height = 1, width = 1.
- For MNIST / Fashion-MNIST: channels=1, height=28, width=28, classes=10.
- For CIFAR-10: channels=3, height=32, width=32, classes=10.
- A valid graph needs at least one inputNode → one or more denseNodes → one outputNode.
- Never use node types outside the table above.
"""


def build_context_block(context: dict) -> str:
    parts: list[str] = []

    graph = context.get("graph")
    if graph:
        n = len(graph.get("nodes", []))
        e = len(graph.get("edges", []))
        parts.append(f"Current graph: {n} nodes, {e} edges.")
        nodes_summary = [
            f"  - {nd.get('type','?')} (id={nd.get('id','?')}, data={nd.get('data',{})})"
            for nd in graph.get("nodes", [])
        ]
        if nodes_summary:
            parts.append("Nodes:\n" + "\n".join(nodes_summary))

    errors = context.get("validationErrors", [])
    if errors:
        parts.append("Validation errors:\n" + "\n".join(f"  - {e}" for e in errors))

    dataset = context.get("datasetInfo")
    if dataset:
        parts.append(
            f"Loaded dataset: {dataset.get('name','?')} — "
            f"{dataset.get('rows','?')} rows, {dataset.get('features','?')} features, "
            f"target: {dataset.get('target','?')}, classes: {dataset.get('classCount','?')}"
        )

    training = context.get("trainingResults")
    if training and training.get("epochs"):
        last = training["epochs"][-1]
        parts.append(
            f"Last training run: {len(training['epochs'])} epochs, "
            f"final val accuracy {last.get('valAccuracy', 0)*100:.1f}%, "
            f"val loss {last.get('valLoss', 0):.4f}."
        )

    return "\n".join(parts) if parts else "No active graph, dataset, or training data."


async def chat(
    messages: list[dict],
    context: dict,
    api_key: str,
    model: str = "gpt-4o",
) -> str:
    try:
        from openai import AsyncOpenAI  # imported lazily so the app starts without it
    except ImportError:
        return (
            "The `openai` Python package is not installed. "
            "Run `pip install openai` in the backend environment, then restart."
        )

    client = AsyncOpenAI(api_key=api_key)
    context_block = build_context_block(context)
    system = SYSTEM_PROMPT + f"\n\n## Current session state\n{context_block}"

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=1500,
            temperature=0.4,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        err = str(exc)
        if "api_key" in err.lower() or "authentication" in err.lower() or "401" in err:
            return "Invalid API key. Check your OpenAI key in the AI panel settings."
        if "model" in err.lower() or "404" in err:
            return f"Model '{model}' not available on your OpenAI account. Try gpt-3.5-turbo."
        return f"OpenAI API error: {err}"
