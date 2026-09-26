// The run flow: create a run, stream it, render the result (D-080, D-081).
//
// htmx handles the history list; this handles the parts it can't. The SSE extension documents
// no way to append streamed tokens (verified 2026-09-21), and a review accumulating token by
// token needs something to hold the buffer -- so the token stream uses EventSource directly.

const form = document.getElementById("ask");
const liveRun = document.getElementById("live-run");
const liveQuestion = document.getElementById("live-question");
const loadedRun = document.getElementById("loaded-run");
const progress = document.getElementById("progress");
const review = document.getElementById("review");
const violations = document.getElementById("violations");
const coverage = document.getElementById("coverage");
const submit = document.getElementById("submit");

const providerSelect = document.getElementById("provider");
const modelSelect = document.getElementById("model");
const customModelRow = document.getElementById("custom-model-row");
const customModelInput = document.getElementById("custom-model");
const CUSTOM = "__custom__";

const followUp = document.getElementById("follow-up");
const followUpQuestion = document.getElementById("follow-up-question");
const followUpSubmit = document.getElementById("follow-up-submit");

// The thread a follow-up would continue, or null when there's nothing on screen to follow up
// on. Set from the `done` event (live run) or from the fragment's data-follow-up (a past
// conversation) -- in both cases by the server having produced a review, never by guessing.
let followUpTarget = null;

// Node name -> what the reader should be told it means.
const NODE_LABELS = {
  intake: "Reading the question",
  decompose: "Planning subtopics",
  // NOT "Searching arXiv": since local-first (D-105) a worker may answer entirely from the
  // corpus, and the `progress` events right above it say so by name. Claiming a search that
  // did not happen is D-118's bug, which was fixed in the worker and left standing here.
  research_worker: "Researching a subtopic",
  gap_check: "Checking for gaps",
  synthesize: "Writing the review",
  // Added by D-113; the label was not, so the trail showed the raw node name to the reader.
  check_claims: "Checking claims against the papers",
  check_citations: "Verifying citations",
};

// --- the model picker (D-126) -------------------------------------------------------------

// Show only the models belonging to the selected provider, and keep "Custom…" always. The
// options carry their own provider, so this needs no second copy of the list in JS -- the
// server rendered it from config, and that stays the only source of truth.
function syncModelsToProvider() {
  const provider = providerSelect.value;
  let firstVisible = null;
  for (const option of modelSelect.options) {
    const belongs = option.value === CUSTOM || option.dataset.provider === provider;
    option.hidden = !belongs;
    option.disabled = !belongs;
    if (belongs && option.value !== CUSTOM && firstVisible === null) firstVisible = option.value;
  }
  // Switching provider while a now-hidden model is selected would submit a model the provider
  // does not serve. Fall back to its first option, which is the configured default.
  const current = modelSelect.selectedOptions[0];
  if (!current || current.disabled) {
    modelSelect.value = firstVisible ?? CUSTOM;
  }
  syncCustomModelVisibility();
}

function syncCustomModelVisibility() {
  const custom = modelSelect.value === CUSTOM;
  customModelRow.hidden = !custom;
  if (custom) customModelInput.focus();
}

// What POST /runs should receive: null means "the provider's default", which the server and
// the agent both understand as "not chosen" (D-126).
function selectedModel() {
  if (modelSelect.value !== CUSTOM) return modelSelect.value || null;
  return customModelInput.value.trim() || null;
}

providerSelect.addEventListener("change", syncModelsToProvider);
modelSelect.addEventListener("change", syncCustomModelVisibility);
syncModelsToProvider();

function setFollowUpTarget(threadId) {
  followUpTarget = threadId;
  followUp.hidden = threadId === null;
  if (threadId === null) followUpQuestion.value = "";
}

function showLiveRun() {
  progress.replaceChildren();
  review.replaceChildren();
  violations.hidden = true;
  coverage.replaceChildren();
  coverage.hidden = true;
  liveQuestion.textContent = "";
  liveQuestion.hidden = true;
  // A live run and a loaded one are never both on screen: two reviews side by side is a
  // good way to misread which one you're looking at.
  loadedRun.replaceChildren();
  liveRun.hidden = false;
  // Nothing to follow up on until this run produces a review.
  setFollowUpTarget(null);
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
    followUpSubmit.disabled = false;
    // Only offer a follow-up once there is an answer to follow up on -- POST /runs rejects
    // the other case with 409 anyway (D-121), and a button that 409s is worse than no button.
    if (data.review) setFollowUpTarget(data.thread_id);
    document.body.dispatchEvent(new Event("refresh-history"));
  });

  // The server got far enough to explain itself -- a bad model name, a provider outage
  // (D-125). Distinct from the `error` handler below, which fires when the connection itself
  // drops and the server said nothing.
  source.addEventListener("failed", (e) => {
    addProgress(`Run stopped: ${JSON.parse(e.data).message}`);
    source.close();
    submit.disabled = false;
    followUpSubmit.disabled = false;
  });

  source.addEventListener("error", () => {
    // EventSource retries on its own, which would silently restart the run. The checkpoint
    // makes reconnecting safe (D-081), but an automatic retry hides failures -- so stop and
    // let the reader decide.
    source.close();
    addProgress("Connection lost - reload to resume from where it stopped");
    submit.disabled = false;
    followUpSubmit.disabled = false;
  });
}

// Create a run and stream it. `followUpTo` is null for a new question and a thread_id for a
// follow-up; the server does the rewriting, so this is the only difference between the two
// (D-121) -- one request shape, one stream, no second code path.
async function startRun(question, provider, followUpTo) {
  showLiveRun();
  markActive(null);

  const response = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // null model means the provider's default (D-126); the server validates the shape and
    // answers 422 rather than starting a run that will fail partway through.
    body: JSON.stringify({
      question,
      provider,
      model: selectedModel(),
      follow_up_to: followUpTo,
    }),
  });

  if (!response.ok) {
    // 422 carries FastAPI's validation detail, which names what was wrong with the model.
    let reason = `${response.status}`;
    try {
      const detail = (await response.json()).detail;
      if (Array.isArray(detail) && detail[0]?.msg) reason = detail[0].msg;
      else if (typeof detail === "string") reason = detail;
    } catch {
      // A non-JSON error body is fine; the status code alone is still useful.
    }
    addProgress(`Could not start the run: ${reason}`);
    submit.disabled = false;
    followUpSubmit.disabled = false;
    return;
  }

  const created = await response.json();
  // For a follow-up this is the rewritten question, not what was typed. Showing it is the
  // whole reason POST /runs returns it: textContent, so it stays a text sink.
  if (followUpTo) {
    liveQuestion.textContent = created.question;
    liveQuestion.hidden = false;
  }
  streamRun(created.thread_id);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submit.disabled = true;
  startRun(
    document.getElementById("question").value,
    document.getElementById("provider").value,
    null,
  );
});

followUp.addEventListener("submit", (event) => {
  event.preventDefault();
  if (followUpTarget === null) return;
  const question = followUpQuestion.value;
  followUpSubmit.disabled = true;
  // Read the target before showLiveRun() clears it: the follow-up's parent is the run that
  // was on screen when it was asked, not whatever is on screen once the new one starts.
  const parent = followUpTarget;
  followUpQuestion.value = "";
  startRun(question, document.getElementById("provider").value, parent);
});


// --- the sidebar -------------------------------------------------------------------------

// Opening a past run hides the live pane, so only one review is ever on screen. htmx has
// already swapped the fragment in by the time this fires.
document.body.addEventListener("htmx:afterSwap", (event) => {
  if (event.target.id !== "loaded-run") return;
  liveRun.hidden = true;
  // The server sets data-follow-up only when the last turn actually has a review (D-121),
  // so the composer appears for exactly the conversations that can be continued.
  const session = event.target.querySelector(".session");
  setFollowUpTarget(session?.dataset.followUp ?? null);
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
  // A new question starts a new conversation: nothing on screen to follow up on.
  setFollowUpTarget(null);
  document.getElementById("question").focus();
});
