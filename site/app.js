// The Little Rhyme Maker : la page du générateur de comptines.
// Parle au serveur local (serveur.py) par POST /api/generate.

const MAX_WORDS = 3;
const MIN_WORDS = 2;
const SLOT_COLORS = ["var(--blue)", "var(--green)", "var(--orange)"];
const PALETTE = ["#0477bd", "#389975", "#ffc500", "#e5644d", "#fc94aa", "#f7914e"];
const BUBBLE_COLORS = ["var(--blue)", "var(--green)", "var(--yellow)", "var(--red)", "var(--pink)", "var(--orange)"];

// Des mots simples, que le modèle connaît bien (TinyStories et comptines).
const SUGGESTIONS = [
  "cat", "dog", "moon", "star", "sun", "sea", "boat", "king", "queen", "pie",
  "cake", "mouse", "cheese", "rain", "garden", "flower", "bird", "tree", "frog", "sheep",
  "cow", "bell", "ship", "snow", "apple", "bear", "fish", "hat", "bread", "horse",
];
const BUBBLES_SHOWN = 10;

// Servie par serveur.py, la page appelle sa propre adresse ; ouverte directement
// depuis le disque (file://), elle appelle le serveur local à son adresse fixe.
const API_URL = location.protocol === "file:" ? "http://127.0.0.1:8000/api/generate" : "/api/generate";

const words = [null, null, null]; // un mot par emplacement, null = vide
let lastWords = null;

const $ = (id) => document.getElementById(id);
const slotsEl = $("slots");
const input = $("word-input");
const hint = $("hint");
const generateBtn = $("generate-btn");
const bubblesEl = $("bubbles");
const resultEl = $("result");
const writingEl = $("writing");
const poemEl = $("poem");
const metaEl = $("meta");
const againBtn = $("again-btn");

const clean = (w) => w.toLowerCase().replace(/[^a-z]/g, "").slice(0, 20);
const filled = () => words.filter(Boolean);

// --- Le titre : chaque lettre saute et prend une couleur au survol -------------------------

function splitTitle() {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelectorAll(".title-line").forEach((line) => {
    const text = line.textContent;
    line.textContent = "";
    for (const ch of text) {
      const span = document.createElement("span");
      span.className = ch === " " ? "char space" : "char";
      span.textContent = ch === " " ? " " : ch;
      span.setAttribute("aria-hidden", "true");
      if (ch !== " " && !reduce) span.addEventListener("pointerenter", () => jump(span));
      line.appendChild(span);
    }
  });
}

function jump(span) {
  if (span.dataset.busy) return;
  span.dataset.busy = "1";
  const color = PALETTE[Math.floor(Math.random() * PALETTE.length)];
  const up = -(12 + Math.random() * 10);
  span.animate(
    [
      { transform: "translateY(0)", color: "#2d2d2d" },
      { transform: `translateY(${up}px)`, color, offset: 0.25 },
      { transform: "translateY(0)", color, offset: 0.6 },
      { transform: "translateY(0)", color: "#2d2d2d" },
    ],
    { duration: 1400, easing: "cubic-bezier(.3,.7,.4,1)" },
  ).onfinish = () => delete span.dataset.busy;
}

// --- Les trois emplacements --------------------------------------------------------

function renderSlots() {
  slotsEl.textContent = "";
  words.forEach((word, i) => {
    const li = document.createElement("li");
    li.className = "slot" + (word ? " filled" : "");
    if (word) {
      const field = document.createElement("input");
      field.value = word;
      field.maxLength = 20;
      field.setAttribute("aria-label", `Word ${i + 1}, click to change it`);
      field.addEventListener("keydown", (e) => { if (e.key === "Enter") field.blur(); });
      field.addEventListener("change", () => setWord(i, field.value));
      const remove = document.createElement("button");
      remove.className = "remove";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${word}`);
      remove.addEventListener("click", () => setWord(i, ""));
      li.append(field, remove);
    } else {
      li.textContent = `word ${i + 1}`;
    }
    slotsEl.appendChild(li);
  });
  updateControls();
}

function setWord(i, raw) {
  const word = clean(raw);
  words[i] = word && !words.some((w, j) => j !== i && w === word) ? word : null;
  renderSlots();
}

function addWord(raw) {
  const word = clean(raw);
  if (!word) return;
  if (words.includes(word)) { say(`"${word}" is already there.`); return; }
  const free = words.indexOf(null);
  if (free === -1) { say("3 words max: click a word to change it, or remove one."); return; }
  words[free] = word;
  renderSlots();
}

function updateControls() {
  const n = filled().length;
  generateBtn.disabled = n < MIN_WORDS;
  input.disabled = n >= MAX_WORDS;
  input.placeholder = n >= MAX_WORDS ? "All set! Click a word above to change it" : "Type a word, then press Enter";
  $("add-btn").disabled = n >= MAX_WORDS;
  say(n < MIN_WORDS ? `Add at least ${MIN_WORDS - n} more word${MIN_WORDS - n > 1 ? "s" : ""}.`
      : n < MAX_WORDS ? "Ready! Add a third word, or generate now." : "Ready to generate.");
  bubblesEl.querySelectorAll(".bubble").forEach((b) => {
    b.classList.toggle("used", words.includes(b.dataset.word));
    b.disabled = n >= MAX_WORDS && !words.includes(b.dataset.word);
  });
}

function say(text) { hint.textContent = text; }

// --- Les bulles d'idées ---------------------------------------------------------------

function renderBubbles() {
  const pool = SUGGESTIONS.filter((w) => !words.includes(w)).sort(() => Math.random() - 0.5);
  bubblesEl.textContent = "";
  pool.slice(0, BUBBLES_SHOWN).forEach((word, i) => {
    const b = document.createElement("button");
    b.className = "bubble";
    b.type = "button";
    b.dataset.word = word;
    b.textContent = word;
    b.style.setProperty("--c", BUBBLE_COLORS[i % BUBBLE_COLORS.length]);
    b.style.animationDelay = `${-Math.random() * 4}s`;
    b.addEventListener("click", () => addWord(word));
    bubblesEl.appendChild(b);
  });
  updateControls();
}

// --- La génération -----------------------------------------------------------------

async function generate(list) {
  lastWords = list;
  resultEl.hidden = false;
  writingEl.hidden = false;
  poemEl.textContent = "";
  metaEl.textContent = "";
  againBtn.hidden = true;
  generateBtn.disabled = true;
  resultEl.scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    let response;
    try {
      response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: list }),
      });
    } catch {
      // Pas de réponse du tout : le serveur n'est pas lancé (ou pas à cette adresse).
      throw new Error("The rhyme server is not answering. In the project folder, run "
        + "\"uv run python serveur.py\", wait for \"site : http://localhost:8000\", then open that address.");
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "the server said no");
    showPoem(data);
  } catch (err) {
    metaEl.textContent = "";
    const p = document.createElement("span");
    p.className = "error";
    p.textContent = `Oops, no comptine this time. ${err.message}`;
    metaEl.appendChild(p);
  } finally {
    writingEl.hidden = true;
    againBtn.hidden = false;
    updateControls();
  }
}

function showPoem(data) {
  const colorOf = (w) => SLOT_COLORS[Math.max(0, words.indexOf(w))];
  const pattern = new RegExp(`\\b(${data.words.join("|")})\\b`, "gi");
  data.lines.forEach((line, i) => {
    const div = document.createElement("div");
    div.className = "poem-line";
    div.style.animationDelay = `${i * 0.35}s`;
    let last = 0;
    for (const m of line.matchAll(pattern)) {
      div.append(line.slice(last, m.index));
      const mark = document.createElement("mark");
      mark.textContent = m[0];
      mark.style.setProperty("--c", colorOf(m[0].toLowerCase()));
      div.append(mark);
      last = m.index + m[0].length;
    }
    div.append(line.slice(last));
    poemEl.appendChild(div);
  });
  const rhymes = Math.round(data.rhyme_rate * data.lines.length);
  metaEl.textContent = `${data.found.length} of ${data.words.length} words used · `
    + `${rhymes} of ${data.lines.length} lines rhyme · written in ${data.seconds} s`;
}

// --- Branchements -----------------------------------------------------------------

$("add-form").addEventListener("submit", (e) => {
  e.preventDefault();
  addWord(input.value);
  input.value = "";
  input.focus();
});
generateBtn.addEventListener("click", () => generate(filled()));
againBtn.addEventListener("click", () => generate(lastWords || filled()));
$("shuffle-btn").addEventListener("click", renderBubbles);

splitTitle();
renderSlots();
renderBubbles();
