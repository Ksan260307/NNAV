# 元記事『【Python】ロックマンエグゼのネットナビを本気で実現する（プロダクト草案）』の
# コアエンジン。比較用にそのまま置いてある。
#
#     python legacy/netnavi_engine.py
#
# 現行版との違いは README の「3. しくみ」を参照。
# (密行列シナプス / 2-gram のみ / 文法制約なし / 単一候補生成 / 毎ターン全書き出し)

import os
import json
import torch
import warnings

# 警告メッセージを沈黙させる
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from janome.tokenizer import Tokenizer
except ImportError:
    print("[警告] 自然言語処理ライブラリ 'janome' がインストールされていません。")
    print("        実行前に `pip install janome` を実行してください。")
    print("        (インストールされていない場合、文字単位の学習モードで動作します)")
    Tokenizer = None

# =====================================================================
# ハードウェア極限最適化 (PyTorch)
# =====================================================================
torch.set_num_threads(os.cpu_count() or 4)

# 非正規数(Denormal)によるCPUの深刻な計算遅延ペナルティをハードウェアレベルで回避
torch.set_flush_denormal(True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# =====================================================================
# ネットナビ・自律成長型コアエンジン
# =====================================================================


class PyUCDF_NaviBrain:
    def __init__(self, max_vocab=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_vocab = max_vocab
        self.vocab_file = "navi_vocab.json"
        self.tensor_file = "navi_tensors.pt"

        self.tokenizer = Tokenizer() if Tokenizer else None

        # --- 語彙（単語ニューロン）の管理 ---
        # 0: 文の始まり (BOS), 1: 文の終わり (EOS)
        self.word2id = {"__BOS__": 0, "__EOS__": 1}
        self.id2word = {0: "__BOS__", 1: "__EOS__"}
        self.vocab_size = 2

        # --- 状態管理テンソル (Structure of Arrays) ---
        # 各要素 (32bit) に [vA(8bit), vB(8bit), vC(8bit), State(3bit), Ruin(3bit)] をパック
        self.state_buffer = torch.zeros(max_vocab, dtype=torch.int32, device=self.device)

        # --- シナプス結合行列 (マルコフ連鎖) ---
        # 単語間の遷移頻度を記録する [max_vocab, max_vocab] の行列
        self.transition_matrix = torch.zeros(
            (max_vocab, max_vocab), dtype=torch.float32, device=self.device)

        # 初期状態パック (BOSとEOSはシステム予約のため、重要度MAXで固定)
        self._pack_state(0, va=255, vb=255, vc=0, state=1, ruin=0)
        self._pack_state(1, va=255, vb=255, vc=0, state=1, ruin=0)

        self.load_memory()

    def _pack_state(self, idx, va, vb, vc, state, ruin):
        """指定したインデックスのニューロン状態を32bitにパック"""
        packed = ((int(va) & 0xFF) |
                  ((int(vb) & 0xFF) << 8) |
                  ((int(vc) & 0xFF) << 16) |
                  ((int(state) & 0x07) << 24) |
                  ((int(ruin) & 0x07) << 27))
        self.state_buffer[idx] = torch.tensor(packed, dtype=torch.int32, device=self.device)

    def save_memory(self):
        """脳の状態（テンソルと語彙辞書）をファイルに保存"""
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump({"word2id": self.word2id}, f, ensure_ascii=False, indent=2)

        # テンソルは必要なサイズだけスライスして保存し、容量を節約
        torch.save({
            "state_buffer": self.state_buffer[:self.vocab_size].cpu(),
            "transition_matrix": self.transition_matrix[:self.vocab_size, :self.vocab_size].cpu()
        }, self.tensor_file)

    def load_memory(self):
        """ファイルから脳の状態を復元"""
        if os.path.exists(self.vocab_file) and os.path.exists(self.tensor_file):
            print("[System] 既存の電脳メモリをロードしています...")
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.word2id = data["word2id"]
                self.id2word = {int(v): k for k, v in self.word2id.items()}
                self.vocab_size = len(self.word2id)

            tensors = torch.load(self.tensor_file, map_location=self.device, weights_only=True)
            saved_vocab_size = tensors["state_buffer"].shape[0]

            self.state_buffer[:saved_vocab_size] = tensors["state_buffer"].to(self.device)
            self.transition_matrix[:saved_vocab_size, :saved_vocab_size] = \
                tensors["transition_matrix"].to(self.device)

            print(f"[System] ロード完了。現在の語彙数: {self.vocab_size - 2}語")
        else:
            print("[System] 新規ニューラルネットワークを初期化。ゼロから学習を開始します。")

    def apply_vif_entropy_decay(self):
        """
        テンソル一括演算による忘却・減衰プロセス
        ループを排除し、数万の単語ニューロンの「忘却」を一瞬で計算します。
        """
        if self.vocab_size <= 2:
            return

        # BOS(0) と EOS(1) を除く実際の単語ニューロンのみ対象
        states = self.state_buffer[2:self.vocab_size]

        va = states & 0xFF            # 重要度 (よく使われる言葉ほど高い)
        vb = (states >> 8) & 0xFF     # 演算頻度
        vc = (states >> 16) & 0xFF    # 減衰カウンタ (時間経過で上昇)
        st = (states >> 24) & 0x07
        ruin = (states >> 27) & 0x07

        # ターン経過ごとに減衰カウンタ(vC)を増加
        vc = torch.clamp(vc + 5, max=255)

        # vCが閾値(100)を超えたら重要度(vA)を削り、カウンタをリセット
        decay_mask = (vc > 100).to(torch.int32)
        va = torch.clamp(va - (decay_mask * 2), min=1)
        vc = torch.where(decay_mask > 0, torch.zeros_like(vc), vc)

        # 再パックして上書き
        packed = (va | (vb << 8) | (vc << 16) | (st << 24) | (ruin << 27))
        self.state_buffer[2:self.vocab_size] = packed

        # シナプス結合（マルコフ行列）の自然減衰
        # 時間経過とともに、使われない言葉のつながりも徐々に薄れ、忘れ去られる
        self.transition_matrix *= 0.99

    def learn_from_input(self, text):
        """ユーザーの入力から単語を抽出し、シナプス（遷移行列）を強化する"""

        # NLP環境フォールバック: Janomeがない場合は文字単位で学習
        if self.tokenizer:
            tokens = [t.surface for t in self.tokenizer.tokenize(text)]
        else:
            tokens = list(text)

        prev_id = 0  # __BOS__
        used_ids = []

        for surface in tokens:
            # 空白や一部の記号は無視
            if surface.strip() == "" or surface in ["、", "・"]:
                continue

            # 未知の単語ならニューロンを新規割り当て
            if surface not in self.word2id:
                if self.vocab_size >= self.max_vocab:
                    prev_id = 0  # 容量限界: 誤った文脈接続を防ぐためリセット
                    continue
                new_id = self.vocab_size
                self.word2id[surface] = new_id
                self.id2word[new_id] = surface
                self.vocab_size += 1

                # 新規ニューロンの初期化 (少し高めの重要度でスタート)
                self._pack_state(new_id, va=100, vb=50, vc=0, state=1, ruin=0)

            curr_id = self.word2id[surface]
            used_ids.append(curr_id)

            # シナプス結合（遷移確率）を強化
            self.transition_matrix[prev_id, curr_id] += 1.0
            prev_id = curr_id

            # 句点などで文が終わったと判定
            if surface in ["。", "！", "？", "!", "?"]:
                self.transition_matrix[prev_id, 1] += 1.0  # __EOS__
                prev_id = 0  # 次の文は再びBOSから

        # 最後に文が終わっていなければEOSへ繋ぐ
        if prev_id != 0:
            self.transition_matrix[prev_id, 1] += 1.0

        # 今回使用された単語ニューロンを一括で「刺激（重要度アップ）」する
        if used_ids:
            idx_tensor = torch.tensor(list(set(used_ids)), dtype=torch.long, device=self.device)
            states = self.state_buffer[idx_tensor]

            va = states & 0xFF
            vb = (states >> 8) & 0xFF
            vc = (states >> 16) & 0xFF
            st = (states >> 24) & 0x07
            ruin = (states >> 27) & 0x07

            # 刺激による回復
            va = torch.clamp(va + 50, max=255)
            vb = torch.clamp(vb + 10, max=255)
            vc = torch.zeros_like(vc)  # 減衰リセット

            packed = (va | (vb << 8) | (vc << 16) | (st << 24) | (ruin << 27))
            self.state_buffer[idx_tensor] = packed.to(torch.int32)

        self.apply_vif_entropy_decay()
        self.save_memory()

    def generate_response(self):
        """確率雲（テンソル）からのサンプリングによる発話生成"""
        if self.vocab_size <= 2:
            return "……（まだ言葉を知りません）", "……"

        # 全ニューロンの重要度(vA)を抽出 (0.0 ~ 1.0 に正規化)
        states = self.state_buffer[:self.vocab_size]
        va_normalized = (states & 0xFF).to(torch.float32) / 255.0

        curr_id = 0  # __BOS__
        chosen_words = []
        raw_tokens = []

        # 最大30単語で打ち切り
        for _ in range(30):
            # 現在の単語からの遷移頻度ベクトルを取得
            freqs = self.transition_matrix[curr_id, :self.vocab_size].clone()

            # 頻度を対数スケールに変換し、過学習（ループ発話）を防ぐ
            logits = torch.log1p(freqs)

            # BOSへの逆流を禁止
            logits[0] = -1e9

            # softmaxで確率分布に変換 (temperatureでランダム性を制御)
            temperature = 0.8
            context_probs = torch.softmax(logits / temperature, dim=0)

            # 突然変異（Mutation）の計算
            # 重要度(vA)に基づく確率分布をブレンドし、脈絡のない話題へのジャンプを実現する
            if freqs.max() == 0:
                mutation_rate = 1.0   # 未知の状態なら100%突然変異
            else:
                mutation_rate = 0.05  # 5%の確率で脳内で気になっている言葉が口走られる

            va_probs = va_normalized.clone()
            va_probs[0] = 0.0  # BOSには飛ばない
            if va_probs.sum() > 0:
                va_probs = va_probs / va_probs.sum()
            else:
                va_probs = context_probs

            # ブレンドされた最終的な確率分布
            probs = context_probs * (1.0 - mutation_rate) + va_probs * mutation_rate

            # テンソル演算によるランダムサンプリング
            next_id = torch.multinomial(probs, 1).item()

            if next_id == 1:  # __EOS__ に到達したら終了
                break

            word = self.id2word[next_id]
            chosen_words.append(word)

            # 内部解析用のタグ付け処理（メタ認知）
            if self.tokenizer:
                # ジェネレーターをリストに変換してアクセスエラーを回避
                tokens = list(self.tokenizer.tokenize(word))
                if tokens:
                    pos = tokens[0].part_of_speech.split(',')[0]
                    if pos in ['名詞', '動詞', '形容詞']:
                        raw_tokens.append(f"[{word}:{pos}]")
                    else:
                        raw_tokens.append(word)
            else:
                raw_tokens.append(word)

            curr_id = next_id

        if not chosen_words:
            return "……", "……"

        return "".join(chosen_words), "".join(raw_tokens)


# =====================================================================
# メイン実行ループ
# =====================================================================

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("   PyUCDF ネットナビ・コアエンジン")
    print("   (ハードウェア最適化モデル: PyTorch)")
    print("========================================")

    engine = PyUCDF_NaviBrain(max_vocab=10000)

    print("\n[System] ネットナビが起動しました。")
    print("[System] 最初は言葉を知りません。何度も話しかけて、文法や言葉を教えてあげてください。")
    print("[System] 「exit」で終了します。")

    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() in ['exit', 'quit', '終了']:
                print("[Navi]: （ログアウトしました）")
                break
            if not user_input.strip():
                continue

            # 1. ユーザーの言葉から学習する
            engine.learn_from_input(user_input)

            # 2. 脳内テンソルから言葉を紡ぎ出す
            plain_text, analyzed_text = engine.generate_response()

            print(f"\n[Navi]: {analyzed_text}")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
