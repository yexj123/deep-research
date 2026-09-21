// The run flow: create a run, stream it, render the result (D-080, D-081).
//
// htmx handles the history list; this handles the parts it can't. The SSE extension documents
// no way to append streamed tokens (verified 2026-09-21), and a review accumulating token by
// token needs something to hold the buffer -- so the token stream uses EventSource directly.

const form = document.getElementById("ask");
const liveRun = document.getElementById("live-run");
const loadedRun = document.getElementById("loaded-run");
const progress = document.getElementById("progress");
const review = document.getElementById("review");
const violations = document.getElementById("violations");
const coverage = document.getElementById("coverage");
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

function showLiveRun() {
  progress.replaceChildren();
  review.replaceChildren();
  violations.hidden = true;
  coverage.replaceChildren();
  coverage.hidden = true;
  // A live run and a loaded one are never both on screen: two reviews side by side is a
  // good way to misread which one you're looking at.
  loadedRun.replaceChildren();
  liveRun.hidden = false;
}

function markActive(button) {
  for (const entry of document.querySelectorAll(".entry")) {
    entry.classList.toggle("active", entry === button);
  }
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

  // Mid-search status a worker pushed itself (O-5). Node completion can't convey this: a
  // worker waiting on arXiv under a 3-second rate limit is the longest silence in a run.
  source.addEventListener("progress", (e) => {
    const message = JSON.parse(e.data).message;
    if (message) addProgress(message);
  });

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
    // Empty when the run lost nothing, so a clean review isn't padded with a list of
    // nothing. Server-rendered and escaped, like the review itself (D-085, O-5).
    if (data.coverage_html) {
      coverage.innerHTML = data.coverage_html;
      coverage.hidden = false;
    }
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
  showLiveRun();
  markActive(null);

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


// --- the sidebar -------------------------------------------------------------------------

// Opening a past run hides the live pane, so only one review is ever on screen. htmx has
// already swapped the fragment in by the time this fires.
document.body.addEventListener("htmx:afterSwap", (event) => {
  if (event.target.id === "loaded-run") liveRun.hidden = true;
});

document.body.addEventListener("click", (event) => {
  const entry = event.target.closest(".entry");
  if (entry) markActive(entry);

  // Resuming an unfinished run: the checkpoint continues from where it stopped rather than
  // restarting, so this costs at most the node that was in flight (D-081).
  const resume = event.target.closest("[data-resume]");
  if (resume) {
    showLiveRun();
    submit.disabled = true;
    streamRun(resume.dataset.resume);
  }
});

document.getElementById("new-run").addEventListener("click", () => {
  liveRun.hidden = true;
  loadedRun.replaceChildren();
  markActive(null);
  document.getElementById("question").focus();
});
