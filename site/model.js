// Le GPT des comptines, en JavaScript : la même chose que comptines.py, pour le navigateur.
//
// - BPETokenizer : le découpage de tiktoken GPT-2, réduit au vocabulaire du notebook 04 ;
// - GPT          : le passage avant du modèle (embeddings, 6 blocs pre-norm, LayerNorm finale,
//                  couche de sortie partagée avec l'embedding), avec un cache des clés et
//                  valeurs : chaque nouveau token ne recalcule pas toute la séquence ;
// - writeComptine : la génération de generer.py (consigne "rhyme:", coup de pouce sur les
//                  mots-clés, arrêt après 4 vers, meilleur de N).
//
// Les poids viennent de exporter_web.py (docs/model/). Fonctionne dans un Web Worker et dans Node.

(function (root) {
  "use strict";

  // --- Demi-précision → float32 ----------------------------------------------------------

  function halfToFloatTable() {
    const table = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const sign = h & 0x8000 ? -1 : 1;
      const exp = (h >> 10) & 0x1f;
      const frac = h & 0x3ff;
      if (exp === 0) table[h] = sign * Math.pow(2, -14) * (frac / 1024);
      else if (exp === 31) table[h] = frac ? NaN : sign * Infinity;
      else table[h] = sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
    return table;
  }

  function loadWeights(manifest, buffer) {
    const half = new Uint16Array(buffer);
    const table = halfToFloatTable();
    const all = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) all[i] = table[half[i]];
    const tensors = {};
    for (const t of manifest.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      tensors[t.name] = all.subarray(t.offset, t.offset + size);
    }
    return tensors;
  }

  // --- Le tokenizer -------------------------------------------------------------------

  const PRETOKEN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const utf8 = new TextEncoder();
  const latin1 = (bytes) => String.fromCharCode(...bytes);

  class BPETokenizer {
    /** @param {{gpt2_ids: number[], eot: number, ranks: string[]}} data  (docs/model/tokenizer.json) */
    constructor(data) {
      this.ranks = data.ranks; // identifiant GPT-2 → octets du token (chaîne latin-1)
      this.rankOf = new Map(data.ranks.map((s, i) => [s, i]));
      this.gpt2Ids = data.gpt2_ids; // indice compact → identifiant GPT-2
      this.toCompact = new Map(data.gpt2_ids.map((id, i) => [id, i]));
      this.eot = data.eot;
      this.vocabSize = data.gpt2_ids.length;
      this.byteIds = [];
      for (let b = 0; b < 256; b++) this.byteIds.push(this.toCompact.get(this.rankOf.get(String.fromCharCode(b))));
      this.cache = new Map();
      this.newlineCounts = new Int32Array(this.vocabSize);
      data.gpt2_ids.forEach((id, i) => {
        if (id < this.ranks.length) this.newlineCounts[i] = this.ranks[id].split("\n").length - 1;
      });
    }

    /** Le BPE d'un morceau : fusionner, tant qu'il y en a, la paire de rang le plus bas. */
    bpe(piece) {
      const known = this.rankOf.get(piece);
      if (known !== undefined) return [known];
      if (this.cache.has(piece)) return this.cache.get(piece);
      let parts = Array.from(piece);
      while (parts.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.rankOf.get(parts[i] + parts[i + 1]);
          if (r !== undefined && r < best) { best = r; at = i; }
        }
        if (at < 0) break;
        parts.splice(at, 2, parts[at] + parts[at + 1]);
      }
      const ids = parts.map((p) => this.rankOf.get(p));
      this.cache.set(piece, ids);
      return ids;
    }

    /** Texte → indices compacts ; un token absent du vocabulaire réduit s'écrit octet par octet. */
    encode(text) {
      const ids = [];
      for (const m of text.matchAll(PRETOKEN)) {
        for (const id of this.bpe(latin1(utf8.encode(m[0])))) {
          const c = this.toCompact.get(id);
          if (c !== undefined) ids.push(c);
          else for (const ch of this.ranks[id]) ids.push(this.byteIds[ch.charCodeAt(0)]);
        }
      }
      return ids;
    }

    /** Indices compacts → texte. */
    decode(ids) {
      const bytes = [];
      for (const i of ids) {
        const id = this.gpt2Ids[i];
        if (id >= this.ranks.length) continue; // <|endoftext|>
        for (const ch of this.ranks[id]) bytes.push(ch.charCodeAt(0));
      }
      return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
    }
  }

  // --- Le modèle ----------------------------------------------------------------------

  /** Y[b][o] = bias[o] + Σ_i W[o][i] X[b][i], pour les B lignes du lot.
   *  Les lignes sont traitées par blocs de 4 : chaque poids W[o][i] est lu une fois pour 4 lignes. */
  function matmul(Y, X, W, bias, B, nIn, nOut) {
    let b = 0;
    for (; b + 4 <= B; b += 4) {
      const x0 = b * nIn, x1 = x0 + nIn, x2 = x1 + nIn, x3 = x2 + nIn;
      for (let o = 0; o < nOut; o++) {
        const w = o * nIn;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let i = 0; i < nIn; i++) {
          const wi = W[w + i];
          s0 += wi * X[x0 + i]; s1 += wi * X[x1 + i]; s2 += wi * X[x2 + i]; s3 += wi * X[x3 + i];
        }
        const c = bias ? bias[o] : 0;
        Y[b * nOut + o] = s0 + c; Y[(b + 1) * nOut + o] = s1 + c;
        Y[(b + 2) * nOut + o] = s2 + c; Y[(b + 3) * nOut + o] = s3 + c;
      }
    }
    for (; b < B; b++) { // les lignes qui restent, une par une
      const x = b * nIn;
      for (let o = 0; o < nOut; o++) {
        const w = o * nIn;
        let s = bias ? bias[o] : 0;
        for (let i = 0; i < nIn; i++) s += W[w + i] * X[x + i];
        Y[b * nOut + o] = s;
      }
    }
  }

  function layerNorm(Y, X, gamma, beta, B, C) {
    for (let b = 0; b < B; b++) {
      const o = b * C;
      let mean = 0;
      for (let i = 0; i < C; i++) mean += X[o + i];
      mean /= C;
      let v = 0;
      for (let i = 0; i < C; i++) { const d = X[o + i] - mean; v += d * d; }
      const inv = 1 / Math.sqrt(v / C + 1e-5);
      for (let i = 0; i < C; i++) Y[o + i] = (X[o + i] - mean) * inv * gamma[i] + beta[i];
    }
  }

  class GPT {
    constructor(manifest, tensors) {
      Object.assign(this, manifest.config);
      this.w = tensors;
      this.headSize = this.n_embd / this.n_head;
    }

    /** Prépare un lot de B séquences : caches des clés et valeurs, et tampons de calcul. */
    start(B) {
      const C = this.n_embd, T = this.block_size;
      this.B = B;
      this.cacheK = Array.from({ length: this.n_layer }, () => new Float32Array(B * T * C));
      this.cacheV = Array.from({ length: this.n_layer }, () => new Float32Array(B * T * C));
      this.x = new Float32Array(B * C);
      this.h = new Float32Array(B * C);
      this.q = new Float32Array(B * C);
      this.k = new Float32Array(B * C);
      this.v = new Float32Array(B * C);
      this.att = new Float32Array(B * C);
      this.tmp = new Float32Array(B * C);
      this.ff = new Float32Array(B * 4 * C);
      this.scores = new Float32Array(T);
      this.logits = new Float32Array(B * this.vocab_size);
    }

    /** Fait avancer les B séquences d'un token (à la position `pos`) ; renvoie les logits si demandé. */
    step(tokens, pos, wantLogits = true) {
      const { B, n_embd: C, n_head: H, headSize: D, block_size: T, w } = this;
      const { x, h, q, k, v, att, tmp, ff, scores } = this;
      for (let b = 0; b < B; b++) {
        const te = tokens[b] * C, pe = pos * C;
        for (let i = 0; i < C; i++) x[b * C + i] = w.tok_emb[te + i] + w.pos_emb[pe + i];
      }
      const scale = 1 / Math.sqrt(D);
      for (let l = 0; l < this.n_layer; l++) {
        const p = `l${l}.`;
        // Communication : self-attention masquée, sur tout le passé gardé dans le cache.
        layerNorm(h, x, w[p + "ln1_w"], w[p + "ln1_b"], B, C);
        matmul(q, h, w[p + "wq"], null, B, C, C);
        matmul(k, h, w[p + "wk"], null, B, C, C);
        matmul(v, h, w[p + "wv"], null, B, C, C);
        const K = this.cacheK[l], V = this.cacheV[l];
        for (let b = 0; b < B; b++) {
          K.set(k.subarray(b * C, (b + 1) * C), (b * T + pos) * C);
          V.set(v.subarray(b * C, (b + 1) * C), (b * T + pos) * C);
          for (let hd = 0; hd < H; hd++) {
            const qo = b * C + hd * D;
            let max = -Infinity;
            for (let t = 0; t <= pos; t++) {
              const ko = (b * T + t) * C + hd * D;
              let s = 0;
              for (let i = 0; i < D; i++) s += q[qo + i] * K[ko + i];
              s *= scale;
              scores[t] = s;
              if (s > max) max = s;
            }
            let sum = 0;
            for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
            for (let i = 0; i < D; i++) att[qo + i] = 0;
            for (let t = 0; t <= pos; t++) {
              const a = scores[t] / sum, vo = (b * T + t) * C + hd * D;
              for (let i = 0; i < D; i++) att[qo + i] += a * V[vo + i];
            }
          }
        }
        matmul(tmp, att, w[p + "wproj"], w[p + "bproj"], B, C, C);
        for (let i = 0; i < B * C; i++) x[i] += tmp[i];
        // Calcul : feed-forward, position par position.
        layerNorm(h, x, w[p + "ln2_w"], w[p + "ln2_b"], B, C);
        matmul(ff, h, w[p + "w1"], w[p + "b1"], B, C, 4 * C);
        for (let i = 0; i < ff.length; i++) if (ff[i] < 0) ff[i] = 0;
        matmul(tmp, ff, w[p + "w2"], w[p + "b2"], B, 4 * C, C);
        for (let i = 0; i < B * C; i++) x[i] += tmp[i];
      }
      if (!wantLogits) return null;
      layerNorm(h, x, w.ln_f_w, w.ln_f_b, B, C);
      matmul(this.logits, h, w.tok_emb, w.lm_bias, B, C, this.vocab_size); // weight tying
      return this.logits;
    }
  }

  // --- Mots-clés et rimes (comme comptines.py) ------------------------------------------------

  const words = (text) => new Set(text.toLowerCase().match(/[a-z]+/g) || []);

  function keywordHits(keywords, text) {
    const found = words(text);
    return keywords.filter((k) => found.has(k)).length / keywords.length;
  }

  function lastWord(line) {
    const all = line.toLowerCase().match(/[a-z']+/g);
    return all ? all[all.length - 1].replace(/^'+|'+$/g, "") : null;
  }

  function rhymeKey(line) {
    const word = lastWord(line);
    if (word === null) return null;
    const silentE = word.length > 2 && word.endsWith("e") && !"aeiouy".includes(word[word.length - 2]);
    const stem = silentE ? word.slice(0, -1) : word;
    const m = stem.match(/[aeiouy]+[^aeiouy]*$/);
    return (m ? m[0] : stem) + (silentE ? "e" : "");
  }

  function rhymeRate(text) {
    const lines = text.split("\n").filter((l) => l.trim());
    if (lines.length < 2) return 0;
    const keys = lines.map(rhymeKey), ends = lines.map(lastWord);
    const rhyming = new Set();
    for (let i = 0; i < lines.length; i++) {
      for (const j of [i + 1, i + 2]) {
        if (j < lines.length && keys[i] && keys[i] === keys[j] && ends[i] !== ends[j]) { rhyming.add(i); rhyming.add(j); }
      }
    }
    return rhyming.size / lines.length;
  }

  function poemScore(keywords, text, nLines, keywordWeight, maxWords) {
    const lines = text.split("\n").filter((l) => l.trim());
    let penalty = lines.length === nLines ? 0 : 1;
    if (maxWords !== null && lines.length) {
      penalty += lines.filter((l) => l.trim().split(/\s+/).length > maxWords).length / lines.length;
    }
    return keywordWeight * keywordHits(keywords, text) + 0.5 * rhymeRate(text) - penalty;
  }

  /** Suivi token par token des mots-clés déjà écrits (KeywordTracker de comptines.py). */
  class KeywordTracker {
    constructor(tokenizer, keywords, maxTokens = 3, maxFailures = 3) {
      this.forms = keywords.map((w) => {
        const cap = w.charAt(0).toUpperCase() + w.slice(1);
        const seqs = [...new Set([w, cap, " " + w, " " + cap])].map((f) => tokenizer.encode(f));
        return seqs.filter((s) => s.length <= maxTokens && tokenizer.decode(s.slice(0, 1)).trim().length >= 2);
      });
      this.progress = this.forms.map((f) => f.map(() => 0));
      this.done = this.forms.map((f) => f.length === 0);
      this.failures = this.forms.map(() => 0);
      this.maxFailures = maxFailures;
    }

    /** Les tokens à favoriser : la suite des mots commencés, sinon le premier token de chaque forme. */
    nextTokens() {
      const boosted = [];
      this.forms.forEach((forms, k) => {
        if (this.done[k]) return;
        const started = forms.flatMap((s, f) => (this.progress[k][f] > 0 ? [s[this.progress[k][f]]] : []));
        boosted.push(...(started.length ? started : forms.map((s) => s[0])));
      });
      return boosted;
    }

    update(token) {
      this.forms.forEach((forms, k) => {
        if (this.done[k]) return;
        let interrupted = false;
        forms.forEach((seq, f) => {
          let p = this.progress[k][f];
          if (seq[p] === token) p += 1;
          else { if (p > 0) interrupted = true; p = seq[0] === token ? 1 : 0; }
          if (p === seq.length) { this.done[k] = true; p = 0; }
          this.progress[k][f] = p;
        });
        if (interrupted && !this.done[k]) {
          this.failures[k] += 1;
          this.done[k] = this.failures[k] >= this.maxFailures;
        }
      });
    }
  }

  // --- La génération ----------------------------------------------------------------------

  /** La k-ième plus grande valeur, en un passage : on garde les k meilleures, triées. */
  function kthLargest(values, k) {
    const top = new Float32Array(k).fill(-Infinity); // ordre décroissant
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v <= top[k - 1]) continue;
      let j = k - 1;
      while (j > 0 && top[j - 1] < v) { top[j] = top[j - 1]; j--; }
      top[j] = v;
    }
    return top[k - 1];
  }

  /** Tire un token : température, coup de pouce, top-k (script 07 du cours). */
  function sample(logits, offset, V, temperature, topK, boosted, boost) {
    const l = new Float32Array(V);
    for (let i = 0; i < V; i++) l[i] = logits[offset + i] / temperature;
    for (const t of boosted) l[t] += boost;
    const threshold = topK ? kthLargest(l, Math.min(topK, V)) : -Infinity;
    let max = -Infinity;
    for (let i = 0; i < V; i++) if (l[i] >= threshold && l[i] > max) max = l[i];
    let sum = 0;
    for (let i = 0; i < V; i++) { l[i] = l[i] >= threshold ? Math.exp(l[i] - max) : 0; sum += l[i]; }
    let r = Math.random() * sum;
    for (let i = 0; i < V; i++) { r -= l[i]; if (r <= 0) return i; }
    return V - 1;
  }

  /** N comptines pour la même consigne, en un lot ; renvoie les textes. */
  function writeTexts(model, tokenizer, keywords, n, s) {
    const prompt = tokenizer.encode(`${s.genre}: ${keywords.join(", ")}\n`);
    const V = tokenizer.vocabSize;
    model.start(n);
    let logits = null;
    for (let p = 0; p < prompt.length; p++) logits = model.step(new Array(n).fill(prompt[p]), p, p === prompt.length - 1);
    const out = Array.from({ length: n }, () => []);
    const done = new Array(n).fill(false);
    const lines = new Array(n).fill(0);
    const trackers = s.keyword_boost ? Array.from({ length: n }, () => new KeywordTracker(tokenizer, keywords)) : null;
    let pos = prompt.length;
    for (let step = 0; step < s.max_new_tokens && pos < model.block_size; step++) {
      const next = [];
      for (let b = 0; b < n; b++) {
        const boosted = trackers ? trackers[b].nextTokens() : [];
        const t = sample(logits, b * V, V, s.temperature, s.top_k, boosted, s.keyword_boost);
        next.push(t);
        if (done[b]) continue;
        if (trackers) trackers[b].update(t);
        out[b].push(t);
        if (t === tokenizer.eot) done[b] = true;
        lines[b] += tokenizer.newlineCounts[t];
        if (lines[b] >= s.n_lines) done[b] = true;
      }
      if (done.every(Boolean)) break;
      logits = model.step(next, pos++, true);
    }
    return out.map((ids) => {
      const cut = ids.indexOf(tokenizer.eot);
      const text = tokenizer.decode(cut >= 0 ? ids.slice(0, cut) : ids).trim();
      return text.split("\n").map((l) => l.trim()).filter(Boolean).slice(0, s.n_lines).join("\n");
    });
  }

  /** Une comptine : meilleur de N, avec les mots repris (même réponse que serveur.py). */
  function writeComptine(model, tokenizer, settings, rawWords) {
    const start = Date.now();
    const keywords = [...new Set(rawWords.map((w) => w.toLowerCase().replace(/[^a-z]/g, "").slice(0, 20)).filter(Boolean))].slice(0, 3);
    const candidates = writeTexts(model, tokenizer, keywords, settings.n, settings);
    const score = (t) => poemScore(keywords, t, settings.n_lines, settings.keyword_weight, settings.max_words);
    const best = candidates.reduce((a, b) => (score(b) > score(a) ? b : a));
    return {
      lines: best.split("\n"),
      words: keywords,
      found: keywords.filter((k) => keywordHits([k], best)),
      rhyme_rate: rhymeRate(best),
      seconds: Math.round((Date.now() - start) / 10) / 100,
    };
  }

  root.RhymeModel = { BPETokenizer, GPT, loadWeights, writeComptine, writeTexts, keywordHits, rhymeRate, rhymeKey };
  if (typeof module !== "undefined") module.exports = root.RhymeModel;
})(typeof self !== "undefined" ? self : globalThis);
