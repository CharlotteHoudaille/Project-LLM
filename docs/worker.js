// Fait tourner le modèle (model.js) en arrière-plan, pour que la page ne se fige pas
// pendant la génération. Messages : {type: "load"} puis {type: "generate", id, words}.

importScripts("model.js");

let model = null;
let tokenizer = null;
let settings = null;

async function fetchWithProgress(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} : ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) postMessage({ type: "progress", ratio: received / total });
  }
  const all = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { all.set(c, offset); offset += c.length; }
  return all.buffer;
}

async function load() {
  const [manifest, tokData, weights] = await Promise.all([
    fetch("model/manifest.json").then((r) => r.json()),
    fetch("model/tokenizer.json").then((r) => r.json()),
    fetchWithProgress("model/weights.bin"),
  ]);
  tokenizer = new RhymeModel.BPETokenizer(tokData);
  model = new RhymeModel.GPT(manifest, RhymeModel.loadWeights(manifest, weights));
  settings = manifest.generation;
}

onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "load") {
      await load();
      postMessage({ type: "ready" });
    } else if (msg.type === "generate") {
      postMessage({ type: "result", id: msg.id, data: RhymeModel.writeComptine(model, tokenizer, settings, msg.words) });
    }
  } catch (err) {
    postMessage({ type: "error", id: msg.id, message: String(err && err.message ? err.message : err) });
  }
};
