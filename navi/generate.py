"""発話生成と再ランキング。

素朴なマルコフ連鎖は「1本引いて、そのまま喋る」ため破綻率が高い。
ここでは
    1. 複数の戦略(文脈生成 / 連想生成 / 模倣 / 質問)で候補を N 本つくり、
    2. 流暢さ・話題の関連性・新規性・文法の 4 軸で採点し、
    3. 最良のものだけを口に出す
という「考えてから喋る」構造にしている。人間らしさの大半はここで決まる。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import torch

from .brain import BOS, EOS, N_SPECIAL, UNK, NaviBrain
from .jp import Tok, can_end, can_follow, is_terminator, trim_to_valid_end
from .parse import Pred, parse

SILENCE = "……"


@dataclass
class GenParams:
    """成長段階から与えられる発話パラメータ。"""
    min_len: int = 1
    max_len: int = 24
    n_candidates: int = 8
    temperature: float = 0.85
    mutation_rate: float = 0.05
    repetition_penalty: float = 1.6
    topic_bonus: float = 1.2          # 話題関連語への加点(文脈追従の強さ)
    allow_mimic: bool = True
    mimic_bias: float = 0.0           # 模倣記憶への下駄(幼い時期ほど大きい)
    mimic_len_penalty: float = 0.0    # 長文の丸ごと再利用への罰(育つほど大きい)
    allow_echo: bool = False          # 入力のオウム返しを許すか
    allow_question: bool = False
    question_rate: float = 0.0
    silence_threshold: float = -8.0   # これ未満なら黙る
    allow_fact: bool = False          # 知っている事実で答えてよいか
    frame_weight: float = 0.0         # 選択選好(格フレーム)の採点重み
    semantic_weight: float = 0.0      # 意味ベクトルによる話題一致の重み
    fact_bias: float = 2.6            # 知っている事実で答えるときの下駄


@dataclass
class Context:
    input_ids: List[int] = field(default_factory=list)
    input_keys: List[int] = field(default_factory=list)
    input_text: str = ""
    recent_texts: List[str] = field(default_factory=list)
    asking: bool = False                 # 入力が問いかけかどうか
    topic_ids: List[int] = field(default_factory=list)   # 連想で広げた話題語
    energy: float = 0.5
    valence: float = 0.0
    # --- 意味層(無ければ None のまま動く) ---------------------------------
    input_preds: List[Pred] = field(default_factory=list)
    frames: Optional[object] = None     # CaseFrames
    embed: Optional[object] = None      # Embedding
    facts: Optional[object] = None      # FactStore
    turn: int = 0


@dataclass
class Candidate:
    ids: List[int]
    text: str
    annotated: str
    strategy: str
    score: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)
    is_question: bool = False


# ---------------------------------------------------------------------------
# サンプリング
# ---------------------------------------------------------------------------
def _sample(scores: Dict[int, float], temperature: float, rng: random.Random,
            device: torch.device) -> int:
    """スコア表(logit)から 1 つ選ぶ。テンソル演算でのランダムサンプリング。"""
    keys = list(scores.keys())
    if len(keys) == 1:
        return keys[0]
    logits = torch.tensor([scores[k] for k in keys], dtype=torch.float32, device=device)
    probs = torch.softmax(logits / max(temperature, 1e-3), dim=0)
    g = torch.Generator(device="cpu")
    g.manual_seed(rng.getrandbits(31))
    pick = int(torch.multinomial(probs.cpu(), 1, generator=g).item())
    return keys[pick]


def _legal(brain: NaviBrain, prev_tok: Optional[Tok], cid: int,
           consec_particles: int) -> bool:
    if cid < N_SPECIAL:
        return False
    return can_follow(prev_tok, brain.tok_of(cid), consec_particles)


def sample_sentence(brain: NaviBrain, ctx: Context, p: GenParams,
                    rng: random.Random, seed_ids: Optional[Sequence[int]] = None
                    ) -> Optional[List[int]]:
    """脳内のシナプスを辿って 1 文を紡ぐ。"""
    dev = brain.device
    out: List[int] = []
    used: Dict[int, int] = {}
    consec_particles = 0
    prev, prev2 = BOS, None
    prev_tok: Optional[Tok] = None

    # 話題語への加点表(入力から連想された語に下駄を履かせ、文脈に引き寄せる)
    topic: Dict[int, float] = {}
    if ctx.topic_ids and p.topic_bonus > 0:
        for i, tid in enumerate(ctx.topic_ids):
            topic[tid] = p.topic_bonus * (1.0 - i / (len(ctx.topic_ids) + 1.0))

    # 種語が指定されていれば、そこから喋り始める
    if seed_ids:
        for sid in seed_ids:
            if sid < N_SPECIAL:
                continue
            tok = brain.tok_of(sid)
            if not can_follow(prev_tok, tok, consec_particles):
                continue
            out.append(sid)
            used[sid] = used.get(sid, 0) + 1
            consec_particles = consec_particles + 1 if tok.pos == "助詞" else 0
            prev2, prev, prev_tok = prev, sid, tok

    while len(out) < p.max_len:
        raw = brain.next_candidates(prev, prev2)
        if not raw:
            break
        raw.pop(BOS, None)
        raw.pop(UNK, None)
        eos_logit = raw.pop(EOS, None)

        scores: Dict[int, float] = {}
        for cid, lg in raw.items():
            if not _legal(brain, prev_tok, cid, consec_particles):
                continue
            s = lg
            n = used.get(cid, 0)
            if n:
                s -= p.repetition_penalty * n   # 同じ語のループを抑える
            if cid in topic:
                s += topic[cid]
            scores[cid] = s

        # 終止の判断
        can_stop = len(out) >= p.min_len and prev_tok is not None and \
            can_end(prev_tok.pos, prev_tok.surface)
        if eos_logit is not None and can_stop:
            if not scores:
                break
            best = max(scores.values())
            if eos_logit >= best - rng.uniform(0.0, 1.2):
                break

        if not scores:
            # 行き止まり。文として閉じられるならそこで終わる
            if can_stop:
                break
            # 閉じられないなら「突然変異」で脳内の気になっている語へ飛ぶ
            jump = _mutation_jump(brain, prev_tok, consec_particles, rng, used)
            if jump is None:
                break
            scores = {jump: 0.0}

        # 突然変異(元記事の 5%): 文脈を無視して重要度の高い語が漏れる
        if rng.random() < p.mutation_rate:
            jump = _mutation_jump(brain, prev_tok, consec_particles, rng, used)
            if jump is not None:
                scores[jump] = max(scores.values()) + 0.2

        nid = _sample(scores, p.temperature, rng, dev)
        tok = brain.tok_of(nid)
        out.append(nid)
        used[nid] = used.get(nid, 0) + 1
        consec_particles = consec_particles + 1 if tok.pos == "助詞" else 0
        prev2, prev, prev_tok = prev, nid, tok
        if is_terminator(tok.surface):
            break

    toks = [brain.tok_of(i) for i in out]
    toks = trim_to_valid_end(toks)
    if len(toks) < max(1, p.min_len):
        return None
    return [brain.word2id[t.surface] for t in toks]


def _mutation_jump(brain: NaviBrain, prev_tok: Optional[Tok], consec: int,
                   rng: random.Random, used: Dict[int, int]) -> Optional[int]:
    """重要度(vA)の高い語からランダムに 1 語選ぶ(文法的に置ける語のみ)。"""
    n = brain.vocab_size
    if n <= N_SPECIAL:
        return None
    imp = brain.importance()[N_SPECIAL:n].clone()
    for wid, c in used.items():
        if N_SPECIAL <= wid < n:
            imp[wid - N_SPECIAL] *= 0.2 ** c
    if float(imp.sum().item()) <= 0:
        return None
    g = torch.Generator(device="cpu")
    g.manual_seed(rng.getrandbits(31))
    for _ in range(6):
        pick = int(torch.multinomial(imp.cpu(), 1, generator=g).item()) + N_SPECIAL
        if can_follow(prev_tok, brain.tok_of(pick), consec):
            return pick
    return None


# ---------------------------------------------------------------------------
# 候補の採点
# ---------------------------------------------------------------------------
def fluency(brain: NaviBrain, ids: Sequence[int]) -> float:
    """遷移確率の対数平均。文として「言い慣れている」ほど高い。"""
    if not ids:
        return -20.0
    seq = [BOS] + list(ids) + [EOS]
    total = 0.0
    for a, b in zip(seq, seq[1:]):
        row = brain.bi.get(a)
        tot = brain.bi_tot.get(a, 0.0)
        if not row or tot <= 0:
            total += -6.0
            continue
        total += math.log(max(row.get(b, 0.0), 1e-3) / tot)
    return total / (len(seq) - 1)


def score_candidate(brain: NaviBrain, ctx: Context, cand: Candidate,
                    p: GenParams) -> Candidate:
    ids = cand.ids
    keys = [i for i in ids if brain.is_content_id(i)]

    flu = fluency(brain, ids)
    rel = brain.relevance(ctx.input_keys, keys)
    # 直近の発話と丸かぶりしていないか(同じことを言い続けない)
    nov = 0.0
    if cand.text in ctx.recent_texts:
        nov -= 3.0
    uniq_ratio = len(set(ids)) / max(1, len(ids))
    nov += 1.2 * (uniq_ratio - 1.0)
    # 内容語が 1 つも無い相槌だけの発話は弱い(が、短い時期は許す)
    substance = 0.6 * math.log1p(len(keys))
    # 長さ: 短すぎ・長すぎにペナルティ
    ideal = (p.min_len + p.max_len) / 2.0
    length = -0.08 * abs(len(ids) - ideal)
    # 問いかけに問いかけで返さない
    if ctx.asking and cand.is_question:
        total_q = -2.2
    else:
        total_q = 0.0
    # 入力のオウム返し
    echo = 0.0
    if cand.text and cand.text == ctx.input_text and not p.allow_echo:
        echo = -4.0
    # 感情の一致(嫌な記憶の語ばかり並べない)
    val = 0.25 * brain.mood_of(keys) * (1.0 if ctx.valence >= 0 else -1.0)

    # --- 意味の軸 ---------------------------------------------------------
    # 選択選好: その述語のその格に、その名詞を置いてよいか
    frame_term = 0.0
    if ctx.frames is not None and p.frame_weight > 0 and ids:
        pmi, n = ctx.frames.score(parse([brain.tok_of(i) for i in ids]))
        if n:
            frame_term = p.frame_weight * pmi * min(1.0, n / 2.0)
    # 話題の一致を、共起ではなく意味ベクトルの近さで測る
    sem = 0.0
    if ctx.embed is not None and p.semantic_weight > 0:
        sem = p.semantic_weight * ctx.embed.similarity_of_sets(ctx.input_keys, keys)

    # 平均対数確率だけだと長い破綻文が有利になるため、総和も混ぜる。
    # 長く喋るほど、その全区間が「言い慣れている」必要がある。
    flu_term = 1.05 * flu + 0.085 * flu * (len(ids) + 1)
    mimic_pen = 0.0
    if cand.strategy.startswith("mimic"):
        mimic_pen = -p.mimic_len_penalty * len(ids)

    total = (flu_term + 2.4 * rel + nov + substance + length + echo + val + mimic_pen
             + frame_term + sem + total_q + cand.detail.get("bias", 0.0))
    cand.score = total
    cand.detail.update({"flu": flu, "rel": rel, "nov": nov, "sub": substance,
                        "len": length, "echo": echo, "val": val, "mim": mimic_pen,
                        "frame": frame_term, "sem": sem})
    return cand


# ---------------------------------------------------------------------------
# 候補生成の各戦略
# ---------------------------------------------------------------------------
def _clip(brain: NaviBrain, ids: Sequence[int], max_len: int) -> List[int]:
    """思い出した発話を、いま出せる長さまで切り詰める。

    幼い時期は max_len が短いので、覚えている文も断片しか言えない。
    成長して max_len が伸びるほど、同じ記憶をより長く再現できるようになる。
    """
    if len(ids) <= max_len:
        return list(ids)
    toks = [brain.tok_of(i) for i in ids[:max_len]]
    toks = trim_to_valid_end(toks)
    return [brain.word2id[t.surface] for t in toks]


def _make_candidate(brain: NaviBrain, ids: List[int], strategy: str,
                    bias: float = 0.0, is_question: bool = False) -> Optional[Candidate]:
    if not ids:
        return None
    text = "".join(brain.id2word[i] for i in ids)
    ann = _annotate(brain, ids)
    return Candidate(ids=ids, text=text, annotated=ann, strategy=strategy,
                     detail={"bias": bias}, is_question=is_question)


def _annotate(brain: NaviBrain, ids: Sequence[int]) -> str:
    """メタ認知表示: ナビが自分の言葉をどう認識しているか。"""
    out = []
    for i in ids:
        w, pos = brain.id2word[i], brain.pos[i]
        if pos in ("名詞", "動詞", "形容詞"):
            out.append(f"[{w}:{pos}]")
        else:
            out.append(w)
    return "".join(out)


def propose_generated(brain: NaviBrain, ctx: Context, p: GenParams,
                      rng: random.Random) -> List[Candidate]:
    out: List[Candidate] = []
    seeds: List[Optional[List[int]]] = [None]
    # 入力に連想で結び付いた語から喋り始める(話題を引き継ぐ)
    for tid, _ in brain.related(ctx.input_keys, topn=3):
        seeds.append([tid])
    # 入力中の内容語そのものから始める(復唱ではなく話題の起点として)
    for k in ctx.input_keys[:2]:
        seeds.append([k])

    for i in range(p.n_candidates):
        seed = seeds[i % len(seeds)]
        ids = sample_sentence(brain, ctx, p, rng, seed_ids=seed)
        c = _make_candidate(brain, ids or [], "gen" if seed is None else "gen+seed")
        if c:
            out.append(c)
    return out


def propose_mimic(brain: NaviBrain, ctx: Context, p: GenParams,
                  rng: random.Random) -> List[Candidate]:
    """過去にオペレーターが言った言葉から、今の文脈に合うものを思い出す。

    「似た発話の、次に実際に来た発話」を最優先で引く。これは
    オペレーター自身の会話の流れを模倣していることになり、
    語彙が少ない時期でも一気に人間らしくなる。
    """
    if not p.allow_mimic or not brain.phrases:
        return []
    out: List[Candidate] = []

    # 0) まったく同じ言葉を過去に言われたことがあるなら、
    #    「そのとき次に続いた言葉」が最も自然な返事になる。
    #    挨拶のように内容語を持たない発話でも文脈をつかめる。
    exact = brain.phrase_by_text.get(ctx.input_text.strip())
    if exact is not None:
        ph = brain.phrases[exact]
        for nxt, w in sorted(ph.next_idx.items(), key=lambda kv: -kv[1])[:2]:
            nph = brain.phrases[nxt]
            c = _make_candidate(brain, _clip(brain, nph.ids, p.max_len), "mimic-exact",
                                bias=p.mimic_bias + 1.7, is_question=nph.is_question)
            if c:
                out.append(c)

    hits = brain.search_phrases(ctx.input_keys, topn=4)
    for idx, sim in hits:
        ph = brain.phrases[idx]
        # 1) その発話の「次に来た発話」
        if ph.next_idx:
            nxt = max(ph.next_idx.items(), key=lambda kv: kv[1])[0]
            nph = brain.phrases[nxt]
            c = _make_candidate(brain, _clip(brain, nph.ids, p.max_len), "mimic-next",
                                bias=p.mimic_bias + 1.6 * sim,
                                is_question=nph.is_question)
            if c:
                out.append(c)
        # 2) その発話そのもの(相槌的な再利用)
        if sim < 0.98 or p.allow_echo:
            c = _make_candidate(brain, _clip(brain, ph.ids, p.max_len), "mimic-echo",
                                bias=p.mimic_bias + 0.8 * sim,
                                is_question=ph.is_question)
            if c:
                out.append(c)
    return out


def _answer_only(brain: NaviBrain, seed: Sequence[int],
                 ids: Sequence[int]) -> List[int]:
    """答えの語に、学習済みの語尾だけを付ける。

    種のあとに別の内容語が続くと「ロック」への答えが「ロックマン」に
    化けてしまう。内容語が現れた時点で打ち切る。
    """
    out = list(seed)
    if list(ids[:len(seed)]) == list(seed):
        for w in ids[len(seed):]:
            if brain.pos[w] not in ("助詞", "助動詞", "記号"):
                break
            out.append(w)
    toks = trim_to_valid_end([brain.tok_of(i) for i in out])
    return [brain.word2id[t.surface] for t in toks]


def propose_fact(brain: NaviBrain, ctx: Context, p: GenParams,
                 rng: random.Random) -> List[Candidate]:
    """知っている事実で問いに答える。

    テンプレートは持たない。答えの語を「種」にして、学習済みの言い回しで
    文に仕立てさせる。うまく続かなければ、答えの語だけを裸で返す
    (どちらが良いかは再ランキングに決めさせる)。
    """
    if not p.allow_fact or ctx.facts is None or not ctx.input_preds:
        return []
    fact = ctx.facts.query(ctx.input_preds, ctx.turn)
    if fact is None:
        return []
    seed, _ = brain.encode(fact.obj)
    if not seed:
        return []
    # 答えは短く。「ロックマンだよ。」で足りるのに語り出さないようにする。
    short = GenParams(**vars(p))
    short.max_len = max(len(seed) + 1, min(p.max_len, len(seed) + 4))
    short.min_len = min(p.min_len, short.max_len)
    short.mutation_rate = 0.0
    short.topic_bonus = 0.0

    out: List[Candidate] = []
    for _ in range(3):
        ids = sample_sentence(brain, ctx, short, rng, seed_ids=seed)
        c = _make_candidate(brain, _answer_only(brain, seed, ids or []),
                            "fact", bias=p.fact_bias)
        if c:
            out.append(c)
    bare = _make_candidate(brain, list(seed), "fact", bias=p.fact_bias * 0.8)
    if bare:
        out.append(bare)
    return out


def propose_question(brain: NaviBrain, ctx: Context, p: GenParams,
                     rng: random.Random) -> List[Candidate]:
    """知らない言葉について聞き返す。

    疑問文の作り方はハードコードせず、オペレーターから学んだ語尾パターン
    (question_tails)を借りて組み立てる。教わっていなければ質問できない。
    """
    if not p.allow_question or not brain.question_tails or not ctx.input_keys:
        return []
    if rng.random() > p.question_rate:
        return []
    # いちばん「知らない」内容語を選ぶ
    target = min(ctx.input_keys, key=lambda w: brain.knowledge_score(w))
    if brain.knowledge_score(target) > 2.2:
        return []
    tails = list(brain.question_tails.items())
    weights = [w for _, w in tails]
    tail = rng.choices([t for t, _ in tails], weights=weights, k=1)[0]
    ids = [target]
    for surface in tail:
        wid = brain.word2id.get(surface)
        if wid is not None:
            ids.append(wid)
    c = _make_candidate(brain, ids, "question", bias=3.0, is_question=True)
    return [c] if c else []


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
def generate(brain: NaviBrain, ctx: Context, p: GenParams,
             rng: Optional[random.Random] = None) -> Candidate:
    rng = rng or random

    # 答えを知っている問いかけには、まず答える。
    # (事実の言い直しは流暢さで勝てないので、他の候補と一緒に並べると
    #  「知っているのに答えない」という一番おかしな振る舞いになる)
    known = propose_fact(brain, ctx, p, rng)
    if known:
        scored = [score_candidate(brain, ctx, c, p) for c in known]
        return max(scored, key=lambda c: c.score)

    # 知らない言葉に出会って好奇心が勝ったターンは、他の候補を捨てて聞き返す。
    questions = propose_question(brain, ctx, p, rng)
    if questions:
        return score_candidate(brain, ctx, questions[0], p)

    cands: List[Candidate] = []
    cands += propose_generated(brain, ctx, p, rng)
    cands += propose_mimic(brain, ctx, p, rng)

    # 同一文の重複を除く
    seen: Set[str] = set()
    uniq: List[Candidate] = []
    for c in cands:
        if c.text in seen:
            continue
        seen.add(c.text)
        uniq.append(score_candidate(brain, ctx, c, p))

    if not uniq:
        return Candidate([], SILENCE, SILENCE, "silence")

    uniq.sort(key=lambda c: -c.score)
    best = uniq[0]
    if best.score < p.silence_threshold:
        return Candidate([], SILENCE, SILENCE, "silence")

    # 上位から少しだけ揺らして選ぶ(常に最善手だと機械的になる)
    top = uniq[: min(3, len(uniq))]
    weights = [math.exp((c.score - top[0].score) * 1.8) for c in top]
    return rng.choices(top, weights=weights, k=1)[0]


def soliloquy(brain: NaviBrain, p: GenParams,
              rng: Optional[random.Random] = None) -> Optional[Candidate]:
    """独り言(擬似睡眠中の自己対話・自発発話)。文脈を持たない。"""
    rng = rng or random
    ctx = Context()
    best: Optional[Candidate] = None
    for _ in range(max(3, p.n_candidates // 2)):
        ids = sample_sentence(brain, ctx, p, rng)
        c = _make_candidate(brain, ids or [], "dream")
        if not c:
            continue
        c = score_candidate(brain, ctx, c, p)
        if best is None or c.score > best.score:
            best = c
    return best
