# How LangGraph behaves, and what its outputs look like

Every output in this document was **captured from this project's own graph**:
`build_graph()` from `src/deep_research/agent/graph.py`, run with
`RecordingFactory` (a fake model that replies `"Attention weighs tokens."`) on
**langgraph 1.2.11 / langchain-core 1.6.3** (2026-09-16). Long IDs are shortened
to `…`. Output formats have changed between minor LangGraph versions, so
capture them again after upgrading.

---

## 0. These are Python objects, not JSON

LangGraph never returns JSON. Its outputs are **Python objects**: `dict`,
`tuple`, your `ResearchState` dataclass, LangChain message objects
(`AIMessageChunk`), and named tuples (`StateSnapshot`). The closest thing to a
"schema" is a set of Python type definitions (`TypedDict`s in `langgraph.types`,
[section 8](#8-the-type-definitions)).

JSON only appears when *you* serialize something, and that can fail:

```text
json.dumps(messages_chunk)  ->  TypeError: Object of type AIMessageChunk is not JSON serializable
```

At milestone 5, the SSE route has to pick out the fields it needs and build its
own JSON.

---

## 1. The execution model in five ideas

1. **State is a set of channels, one per field.** `ResearchState` has two:
   `question` and `review`.
2. **A node reads the state and returns a partial update**, a dict of only the
   keys it changes. Each returned key is written into its channel: through the
   reducer if the field has one, otherwise by replacing the old value.
   Modifying `state` in place writes nothing.
3. **The graph runs in steps (supersteps).** In each step, every scheduled node runs
   (in parallel if there are several), their updates are applied together, and then the
   edges decide what runs next.
4. **A checkpoint is saved after every step** when a checkpointer is attached,
   keyed by `thread_id`.
5. **Streaming is watching that loop from outside.** Each `stream_mode` is a
   different view of the same run.

Your graph's run, step by step (from the `checkpoints` output in §4):

| step | What happened | State afterwards | `next` |
|---|---|---|---|
| -1 | Input received, nothing applied yet | `{}` | `('__start__',)` |
| 0 | `__start__` wrote the input into the channels | `question='  What is attention?  '` | `('intake',)` |
| 1 | `intake` ran and returned `{"question": <stripped>}` | `question='What is attention?'` | `('synthesize',)` |
| 2 | `synthesize` ran and returned `{"review": ...}` | `question=…, review='Attention weighs tokens.'` | `()` (finished) |

That's why token metadata later shows `langgraph_step: 2`: `synthesize` runs in step 2.

---

## 2. `invoke` / `ainvoke`: the final state

| Call | Returns |
|---|---|
| `await graph.ainvoke(input, config, context=ctx)` (v1, default) | `dict` |
| `await graph.ainvoke(..., version="v2")` (this project, D-030) | `GraphOutput` with `.value` and `.interrupts` |

```python
# v1
{'question': 'What is attention?', 'review': 'Attention weighs tokens.'}

# v2
GraphOutput(
    value=ResearchState(question='What is attention?', review='Attention weighs tokens.'),
    interrupts=(),   # human-in-the-loop pauses; always empty for this graph
)
```

---

## 3. The wrapper around each streamed chunk

Every chunk from `astream` comes in one of three wrappers, depending on how you call it:

| How you call `astream` | Each chunk is |
|---|---|
| `stream_mode="updates"` (a single string), v1 | the data itself |
| `stream_mode=["updates", "messages"]` (a list), v1 | a tuple: `(mode, data)` |
| any `stream_mode`, **`version="v2"`** (this project) | a dict: `{"type", "ns", "data"}`, plus `"interrupts"` on `values` chunks |

```python
# v1, single mode
{'intake': {'question': 'What is attention?'}}

# v1, list of modes
('updates', {'intake': {'question': 'What is attention?'}})

# v2
{'type': 'updates', 'ns': (), 'data': {'intake': {'question': 'What is attention?'}}}
```

- **`type`**: which stream mode produced the chunk. Branch on this.
- **`ns`** (namespace): `()` means the chunk came from the top-level graph. It only
  becomes non-empty if you stream from subgraphs.
- **`data`**: the payload. It depends on the mode; see below.

---

## 4. What `data` holds in each stream mode

### `values`: the full state after each step

```python
{'type': 'values', 'ns': (), 'data': ResearchState(question='  What is attention?  ', review=''), 'interrupts': ()}
{'type': 'values', 'ns': (), 'data': ResearchState(question='What is attention?', review=''), 'interrupts': ()}
{'type': 'values', 'ns': (), 'data': ResearchState(question='What is attention?', review='Attention weighs tokens.'), 'interrupts': ()}
```

- There's one chunk per step from step 0 on. The first chunk is the **raw input**,
  spaces included, because `intake` hasn't run yet.
- **v1 gives dicts, and a field that hasn't been written yet is missing, not set to its
  default.** The first v1 chunk was `{'question': '  What is attention?  '}`, with no
  `review` key. v2 converts the dict into `ResearchState`, so the dataclass default
  `review=''` appears.

### `updates`: what each node returned

```python
{'type': 'updates', 'ns': (), 'data': {'intake': {'question': 'What is attention?'}}}
{'type': 'updates', 'ns': (), 'data': {'synthesize': {'review': 'Attention weighs tokens.'}}}
```

- `data` is `{node_name: exactly the dict that node returned}`. The key is the name
  you passed to `add_node`, which is why node names are part of the streaming contract.
- **Not yet checked for this project:** how updates look when several `Send` workers
  finish in the same step. Capture that at milestone 3 before relying on it.

### `messages`: LLM tokens as they arrive

`data` is a **tuple of `(message_chunk, metadata)`**. The fake reply streamed as 5
chunks: `'Attention'`, `' '`, `'weighs'`, `' '`, `'tokens.'`. The fake model splits on
spaces; real models split on tokens.

```python
{'type': 'messages', 'ns': (), 'data': (
    AIMessageChunk(content='Attention', id='lc_run--01a0…', ...),
    {'thread_id': 't', 'langgraph_node': 'synthesize', 'langgraph_step': 2, ...},
)}
```

**The message chunk** (`message_chunk.model_dump()`):

| Field | Captured value | Meaning |
|---|---|---|
| `content` | `'Attention'` | **the text of this chunk.** Joining every chunk's `content` gives the whole reply. |
| `id` | `'lc_run--01a0…'` | **the same for every chunk of one model call.** Use it to group chunks when several calls stream. |
| `type` | `'AIMessageChunk'` | message class |
| `response_metadata` | `{}` | provider details (e.g. finish reason, model name). Empty for the fake model. |
| `usage_metadata` | `None` | token counts, when the provider reports them. `None` for the fake model. |
| `tool_calls`, `tool_call_chunks`, `invalid_tool_calls` | `[]` | tool-calling data (unused here) |
| `additional_kwargs` | `{}` | extra provider-specific fields |
| `name` | `None` | optional speaker name |
| `chunk_position` | `None` | marks the position of a chunk in the stream (e.g. the last one) |

`response_metadata` and `usage_metadata` were **not captured with a real
provider**. Record them the first time your integration test runs against OpenAI.

**The metadata dict** (the full captured contents):

| Key | Captured value | Meaning |
|---|---|---|
| `langgraph_node` | `'synthesize'` | **which node made the LLM call.** This is the key to filter on. |
| `langgraph_step` | `2` | the step the node ran in (see §1) |
| `thread_id` | `'t'` | the run's thread |
| `langgraph_triggers` | `('branch:to:synthesize',)` | what scheduled this node (the edge from `intake`) |
| `langgraph_path` | `('__pregel_pull', 'synthesize')` | internal task path |
| `langgraph_checkpoint_ns`, `checkpoint_ns` | `'synthesize:fdc8…'` | namespace for this one execution of the node |
| `ls_provider` | `'genericfakechatmodel'` | model integration name (the real provider name with `ChatOpenAI`) |
| `ls_model_type` | `'chat'` | kind of model |
| `ls_integration` | `'langchain_chat_model'` | the call went through a LangChain chat model |
| `lc_versions` | `{'langchain-core': '1.6.3'}` | library versions |

**Why tokens stream even though `synthesize` calls `ainvoke`, not `astream`.**
LangGraph hooks into LangChain's callback system. When the `messages` mode is
requested, each chat-model call inside a node streams its tokens out, whichever
method the node used.

**This affects later milestones:** every LLM call in *every* node streams here. Once
`decompose` and `gap_check` call models (milestone 3 onward), their tokens will mix with
`synthesize`'s. Filter on `metadata["langgraph_node"] == "synthesize"`.

### `custom`: whatever your node sends

`custom` chunks only exist if a node calls `get_stream_writer()`. Your graph doesn't
yet, so this was captured from a one-node demo graph:

```python
writer = get_stream_writer()
writer({"status": "searching", "found": 3})
writer("a plain string works too")
```

```python
{'type': 'custom', 'ns': (), 'data': {'status': 'searching', 'found': 3}}
{'type': 'custom', 'ns': (), 'data': 'a plain string works too'}
{'type': 'updates', 'ns': (), 'data': {'emit': {'n': 1}}}
```

- `data` is exactly the object you passed to `writer(...)`, unchanged.
- Chunks are sent **the moment `writer` is called**, before the node finishes and
  before its `updates` chunk.
- **Rule for later:** only pass JSON-serializable data (dicts, strings, numbers)
  to `writer`. That way the SSE route can forward it directly.

### `tasks`: nodes starting and finishing

Two chunk shapes, linked by the same `id`:

```python
# node started
{'type': 'tasks', 'ns': (), 'data': {'id': 'b3d4…', 'name': 'intake',
    'input': ResearchState(question='  What is attention?  ', review=''),
    'triggers': ('branch:to:intake',)}}

# node finished
{'type': 'tasks', 'ns': (), 'data': {'id': 'b3d4…', 'name': 'intake',
    'error': None, 'result': {'question': 'What is attention?'}, 'interrupts': []}}
```

`result` is the dict the node returned. On failure, `error` holds the error.

### `checkpoints`: the saved snapshot after each step

```python
{'type': 'checkpoints', 'ns': (), 'data': {
    'config': {'configurable': {'checkpoint_ns': '', 'thread_id': 't', 'checkpoint_id': '1f1b…41'}},
    'parent_config': {'configurable': {..., 'checkpoint_id': '1f1b…40'}},   # None for step -1
    'values': ResearchState(question='What is attention?', review=''),        # {} at step -1
    'metadata': {'source': 'loop', 'step': 1, 'parents': {}},                 # source 'input' at step -1
    'next': ['synthesize'],
    'tasks': [{'id': 'a30c…', 'name': 'synthesize', 'interrupts': (), 'state': None}],
}}
```

Four chunks, one per step (-1, 0, 1, 2), which is the timeline in §1. `checkpoint_id`
identifies one saved snapshot, and `parent_config` points to the one before it, so the
snapshots form a chain.

### `debug`: `checkpoints` and `tasks` combined

```python
{'type': 'debug', 'ns': (), 'data': {
    'step': 1,
    'timestamp': '2026-09-16T10:09:37.021226+00:00',
    'type': 'task_result',          # 'checkpoint' | 'task' | 'task_result'
    'payload': {...},               # the same shape as the checkpoints / tasks data above
}}
```

8 chunks for this graph. Useful when you're lost; too noisy for the UI.

---

## 5. The order chunks arrive in

Captured with `stream_mode=["values", "updates", "messages"], version="v2"`:

```text
#1      values    input state                       (step 0)
#2      updates   {'intake': {...}}                 (step 1 finished)
#3      values    state after intake
#4–#8   messages  5 tokens while synthesize runs    (step 2)
#9      updates   {'synthesize': {...}}             (step 2 finished)
#10     values    final state
```

**The rule within a step:** tokens (`messages`) and `custom` chunks arrive *while* the
node runs, its `updates` chunk arrives *when it finishes*, and the `values` chunk arrives
once all of the step's updates have been applied.

---

## 6. Reading saved state: `aget_state` and `aget_state_history`

`await graph.aget_state(config)` returns a `StateSnapshot`, a named tuple:

```python
StateSnapshot(
    values={'question': 'What is attention?', 'review': 'Attention weighs tokens.'},
    next=(),
    config={'configurable': {'thread_id': 'snap', 'checkpoint_ns': '', 'checkpoint_id': '1f1b…b4'}},
    metadata={'source': 'loop', 'step': 2, 'parents': {}},
    created_at='2026-09-16T10:09:37.027263+00:00',
    parent_config={'configurable': {'thread_id': 'snap', 'checkpoint_ns': '', 'checkpoint_id': '1f1b…4e'}},
    tasks=(),
    interrupts=(),
)
```

| Field | Meaning |
|---|---|
| `values` | the state. **Always a plain `dict`, even with v2**, and fields that were never written are missing. |
| `next` | nodes still to run. `()` means the run finished. |
| `config` | the config that points to this exact checkpoint (includes `checkpoint_id`) |
| `metadata` | `source` (`'input'` or `'loop'`) and `step` |
| `created_at` | when the checkpoint was saved (ISO timestamp string) |
| `parent_config` | the previous checkpoint |
| `tasks` | pending tasks. After a failure, these hold the error (§7). |
| `interrupts` | pending human-in-the-loop pauses |

`aget_state_history(config)` yields every snapshot, **newest first**:

```text
step= 2 source='loop'  next=()               values={'question': 'What is attention?', 'review': 'Attention weighs tokens.'}
step= 1 source='loop'  next=('synthesize',)  values={'question': 'What is attention?'}
step= 0 source='loop'  next=('intake',)      values={'question': '  What is attention?  '}
step=-1 source='input' next=('__start__',)   values={}
```

**An unknown `thread_id` doesn't raise an error.** It returns an empty snapshot:

```python
StateSnapshot(values={}, next=(), config={'configurable': {'thread_id': 'never-used'}},
              metadata=None, created_at=None, parent_config=None, tasks=(), interrupts=())
```

**At milestone 5,** a `GET /runs/{thread_id}` route must check for this case
(e.g. `created_at is None`) and return 404. Otherwise it returns an empty review
instead of "not found."

---

## 7. Errors

| Situation | What you get (captured) |
|---|---|
| A node raises | **the original exception, unwrapped**: `ValueError: intake: the research question is empty or only whitespace.` |
| A checkpointer is attached but the config has no `thread_id` | `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id` |
| A node returns something that isn't a dict | `InvalidUpdateError: Expected dict, got …` |
| Two nodes write the same key in one step, with no reducer | `InvalidUpdateError: At key 'x': Can receive only one value per step. Use an Annotated key to handle multiple values.` |

**What's saved after a failure.** With a blank question, the last checkpoint stays
at step 0:

```python
next   = ('intake',)
values = {'question': '   '}
tasks  = (PregelTask(name='intake', error="ValueError('intake: the research question is empty or only whitespace.')", result=None, ...),)
```

The failed node is still listed as the next node to run, with its error recorded. This
is what lets LangGraph resume a run from the checkpoint instead of starting over
(documented behavior; not tried in this project yet).

---

## 8. The type definitions

These are the `TypedDict`s LangGraph declares for v2 chunks, captured from `langgraph.types`:

| Type | `type` | `data` |
|---|---|---|
| `ValuesStreamPart` | `'values'` | the state type (also has `interrupts: tuple[Interrupt, ...]`) |
| `UpdatesStreamPart` | `'updates'` | `dict[str, Any]` |
| `MessagesStreamPart` | `'messages'` | `tuple[<a LangChain message or message chunk>, dict[str, Any]]` |
| `CustomStreamPart` | `'custom'` | `Any` |
| `TasksStreamPart` | `'tasks'` | `TaskPayload \| TaskResultPayload` |
| `CheckpointStreamPart` | `'checkpoints'` | `CheckpointPayload` |
| `DebugStreamPart` | `'debug'` | `DebugPayload` |

Every chunk type also has `ns: tuple[str, ...]`. `GraphOutput` (from `ainvoke(..., version="v2")`)
has `value` and `interrupts`. `StreamPart` is the umbrella name for all of the chunk types.

---

## 9. What this means for the project

| Where | What it relies on |
|---|---|
| `test_review_streams_token_by_token_from_synthesize` | `messages` → `data` is `(chunk, metadata)`; `metadata["langgraph_node"]` |
| `test_updates_arrive_in_node_order` | `updates` → the keys of `data` are node names, one chunk per node |
| `test_final_state_is_a_research_state` | `ainvoke(version="v2").value` is a `ResearchState` |
| `test_checkpoint_contains_the_review` | `aget_state(config).values` is a plain dict, looked up by `thread_id` |
| Milestone 5 SSE route | `updates` keys → progress; `messages` `content` filtered by node → the review text; `custom` → forwarded as is. Nothing can be passed to `json.dumps` directly. |
| Milestone 5 "get run" route | an unknown `thread_id` returns an empty snapshot, not an error |

`values` repeats information you already get from `updates`. That's why the
planned SSE route streams `updates`, `custom` and `messages`, but not `values`.

---

## 10. Explore it yourself

Add this inside any async test or script to see raw chunks:

```python
async for chunk in graph.astream(
    {"question": "What is attention?"},
    {"configurable": {"thread_id": "explore"}},
    context=RunContext(provider="openai"),
    stream_mode=["updates", "messages", "values"],
    version="v2",
):
    print(chunk["type"], chunk["ns"], repr(chunk["data"])[:200])
```

---

## Check your understanding

1. Why does the first `values` chunk still have spaces around the question?
2. At milestone 3, the review pane shows planner tokens mixed into the review. Which field fixes that?
3. `aget_state` returns `values={}` and raises no error. What does that tell you, and how can your code detect it?
4. Why can't the SSE route just `json.dumps(chunk)`?
5. In v1, why could `snapshot.values["review"]` raise `KeyError` partway through a run?

<details>
<summary>Answers</summary>

1. That chunk is the state after step 0, when only `__start__` has written the input.
   `intake`, which strips the spaces, runs in step 1.
2. `metadata["langgraph_node"]`: keep only chunks from `"synthesize"`.
3. No run was ever saved under that `thread_id`: it was never used, or you passed a different
   one than the run used. Check `created_at is None` (or `metadata is None`); a real snapshot
   always has both.
4. `data` contains Python objects (`AIMessageChunk`, `ResearchState`) that the `json`
   module can't serialize. The route has to pick fields and build its own JSON.
5. `values` only contains fields that have been written. `review` doesn't exist until
   `synthesize` finishes in step 2.

</details>
