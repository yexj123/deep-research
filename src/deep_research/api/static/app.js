// The run flow: create a run, stream it, render the result (D-080, D-081).
//
// htmx handles the history list; this handles the parts it can't. The SSE extension documents
// no way to append streamed tokens (verified 2026-09-21), and a review accumulating token by
// token needs something to hold the buffer -- so the token stream uses EventSource directly.

const form = document.getElementById("ask");
const progressPanel = document.getElementById("progress-panel");
const progress = document.getElementById("progress");
const reviewPanel = document.getElementById("review-panel");
const review = document.getElementById("review");
const violations = document.getElementById("violations");
const submit = document.getElementById("submit");

// Node name -> what the reader should be told it means.
const NODE_LABELS = {
  intake: "Reading the question",
  decompose: "Planning subtopics",
  research_worker: "Searching arXiv",
  gap_check: "Checking for gaps",
  synthesize: "Writing the review",
  check_citations: "Verifying citations",
};

function reset() {
  progress.replaceChildren();
  review.replaceChildren();
  violations.hidden = true;
  progressPanel.hidden = false;
  reviewPanel.hidden = false;
}

function addProgress(node) {
  const li = document.createElement("li");
  // textContent, not innerHTML: node names come from the server, but this stays a text sink
  // by construction so it can never become an injection point.
  li.textContent = NODE_LABELS[node] || node;
  progress.append(li);
}

function showViolations(ids) {
  if (!ids || ids.length === 0) return;
  violations.textContent =
    `${ids.length} citation(s) could not be verified against the retrieved papers: ` +
    ids.join(", ");
  violations.hidden = false;
}

function streamRun(threadId) {
  const source = new EventSource(`/runs/${threadId}/stream`);
  let buffer = "";

  source.addEventListener("node", (e) => addProgress(JSON.parse(e.data).node));

  source.addEventListener("token", (e) => {
    // Plain text while streaming: no HTML is parsed on this path at all, so a token can
    // never inject anything. The formatted version arrives with `done` (O-8).
    buffer += JSON.parse(e.data).text;
    review.textContent = buffer;
  });

  source.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    // Already rendered AND escaped by the server (api/rendering.py). The browser never
    // parses markdown, so there is no client-side sanitizer to get wrong.
    review.innerHTML = data.review_html;
    showViolations(data.citation_violations);
    source.close();
    submit.disabled = false;
    document.body.dispatchEvent(new Event("refresh-history"));
  });

  source.addEventListener("error", () => {
    // EventSource retries on its own, which would silently restart the run. The checkpoint
    // makes reconnecting safe (D-081), but an automatic retry hides failures -- so stop and
    // let the reader decide.
    source.close();
    addProgress("Connection lost - reload to resume from where it stopped");
    submit.disabled = false;
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  reset();

  const response = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: document.getElementById("question").value,
      provider: document.getElementById("provider").value,
    }),
  });

  if (!response.ok) {
    addProgress(`Could not start the run (${response.status})`);
    submit.disabled = false;
    return;
  }
  streamRun((await response.json()).thread_id);
});
