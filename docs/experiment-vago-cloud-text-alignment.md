# VAGO Cloud入力を文字単位で揃える

Status: OpenRouter 10-seed run complete; physical-T4 runs pending.

## なぜ必要か

これまでのCloud V4は、ViZDoom labelから最接近敵一体の`v / x / a`だけを受け取った。
VAGOの公開Cloud baselineは、画面全体を40×25のbrightness ASCIIにし、同じ40×25位置へ
0〜9のdepth数字を並べる。前者は19文字前後、後者はwrapper込みで毎観測`2,095`文字である。

したがって既存の4.0 killだけでは、architectureの効果と「入力を簡単にした効果」が混ざる。
このadapterはVAGOのCloud入力を同じ35 Hz非停止clockへ載せ、そこを分離する。

## 公開実装から確定した二種類のdepth

- VAGO 1.3M specialist: ASCII token一文字ごとに0〜15のdepth embeddingを合わせる。
- VAGO Cloud LLMAgent: brightness ASCIIとは別に、0〜9を文字として並べたdepth gridを送る。

「全agentがASCII＋depth」という説明は正しいが、specialistとCloudでdepthの表現は同一ではない。
今回揃えるのは比較対象である後者のCloud contractである。

## 実装

- [`vago_text.py`](../src/thought_leak_range/vago_text.py)
  - 公開`AsciiConverter.convert_simple`と同じblock average、brightness charset、40×25 layout
  - 公開`LLMAgent`と同じ0〜9 depth normalization
  - 最終非空行を`shoot / move_forward / turn_left / turn_right`の順でsubstring parse
  - invalid outputは`move_forward`へfallbackする公開挙動も維持
- [`manual_vago_cloud_text_async.py`](../tests/manual_vago_cloud_text_async.py)
  - upstream checkoutから`LLMAgent.SYSTEM_PROMPT`をASTで文字列として取り出す
  - upstream module自体はimportせず、副作用を起こさない
  - 同じ入力を1または3 physical T4、またはOpenRouterへ送る
  - ViZDoomは専用threadで35 Hzを維持し、全laneがbusyなら古い観測をqueueせず捨てる
- [`remote_lane_server.py`](../colab/remote_lane_server.py)
  - 既存の短いV4 endpointとは別に`/vago-text`を追加
  - 完成したaction textを公開parserで解釈してから返す
  - `wire / server compute / queue / prompt token / completion token`を分離記録

system promptをrepositoryへ複製しないのは、比較時点のupstream promptをbyte-exactに使うためでもある。
2026-08-23のupstream READMEはApache 2.0と記す一方、commit `b4c3fdf`のtreeには
`LICENSE` fileも`pyproject.toml`のlicense fieldもないため、adapter側では再配布を避けた。

## ローカル照合

公開upstream commit `b4c3fdf`の`AsciiConverter`をauthoritative implementationとして、
seed 478のランダムRGB/depthを次の16条件で直接比較した。

- 640×480を8 frame
- 320×240を8 frame
- ASCII、改行、code fence、0〜9 depthを含むuser content全体

結果は`byte_exact_cases=16 status=PASS`。加えてproject testは97件すべて通過した。

## 実験行列

同じseed 7〜16、60秒、35 Hz、観測/pulseとも4 ticを基本にする。

| 条件 | 入力 | backend | lane | 目的 |
| --- | --- | --- | ---: | --- |
| S1 | VAGO Cloud text | T4 | 1 | 長いprefillを含む単一GPU基準 |
| S3 | VAGO Cloud text | T4×3 | 3 | 物理分散によるcadence差 |
| OR3 | VAGO Cloud text | OpenRouter | 3 | 論文型Cloud入力＋V4 clock |
| V4-3 | `v / x / a` | T4×3 | 3 | 同じmodel/backendで入力難度だけを戻す |

VAGO公開Cloud loopとの完全なclock比較には、さらに同じLlama 3.1 8B・同じseedを同期loopでも
走らせる。scoreだけでなく、一判断のlatency、action age、判断到着間隔、busy中に捨てた観測数を
必ず併記する。

## すでに見えている面白い予測

短いV4は固定prefixをGPU KV cacheへ置き、変化部分だけを約107 msで処理できた。VAGO Cloud入力は
毎回2,095文字のほぼ全体が変わるため、この近道を使えない。三T4は一判断を三分の一にはしないが、
長いprefillを三台で交代処理できる。

つまりこの比較は単なる「見え方を公平にする」だけでなく、**観測表現そのものがリアルタイム
architectureの一部である**ことも測る。

## OpenRouter配線試験 — 2026-08-23

正式な10×60秒比較の前に、Llama 3.1 8B / Groq固定 / 3 lane / seed 7を15秒だけ実行した。
`max_new_tokens=16`のsmoke条件なので公開benchmarkの集計には混ぜない。

| 指標 | 結果 |
| --- | ---: |
| kill | 1 |
| native clock | 34.28 Hz / valid |
| completed request | 94 |
| gameへ適用 | 85 |
| mean API completion | 378.9 ms |
| mean observation-to-result | 407.4 ms |
| mean action age | 13.48 tic |
| neutral | 246 / 525 tic |
| busyで捨てた観測 | 38 |
| prompt token | mean 771.5、range 667〜897 |
| parser fallback | 0 |
| OpenRouter reported cost | $0.00364529 |

94 responseはすべて公開action grammarへ収まり、provider fallbackも起きなかった。短い構造化V4の
既存平均232.8 msに対し、同じmodel/providerでも完成actionまでは378.9 msだった。まだ一seedの
配線試験だが、入力表現のcostが約146 ms増えた形で観測できた。

生ログは[`vago-cloud-text-openrouter-groq-smoke-seed07-20260823.json`](results/vago-cloud-text-openrouter-groq-smoke-seed07-20260823.json)
（Git LF blob SHA-256 `6f427e7ad3d1f7f168d3a0c257ff7ea81f75ebf9c028b7096c1a161410a38ea6`）。

最初の試行はAPI requestを一件も出す前に、ViZDoomが日本語を含むscenario絶対pathを読めず
`UnicodeDecodeError`で停止した。cfgとWADを内容無変更で`C:\latency-kills-scenario-20260823`へ
複製して解消した。Windows再現手順ではASCII-only pathを要件にする。

## OpenRouter本走行 — 10 episode

seed 7〜16、最大60秒、3 lane、Groq固定、provider fallbackなしで実行した。公開VAGO Cloud
prompt/action parserを使い、推論中もworldを35 Hzで進めた。最初のseed 14だけ31.62 Hzとなり
clock validityを落としたため、同seedを同条件で再走した。再走は34.59 Hzでvalid、killは元runと
同じ1だった。下表はseed 14を再走値へ置換した10本である。

| 指標 | 10 episode |
| --- | ---: |
| kill | **7 / 平均0.7** |
| seed別 | 0, 1, 0, 0, 1, 1, 2, 1, 0, 1 |
| valid clock | **10 / 10** |
| mean native clock | 34.41 Hz |
| mean survival | 13.37 s |
| completed decision | 958 |
| gameへ適用 | 890 |
| mean model/API completion | 360.6 ms |
| mean observation-to-result | 386.3 ms |
| mean action age | 11.75 tic |
| mean neutral | 157.9 tic / episode |
| mean busy-drop | 21.5 observation / episode |
| invalid-output fallback | **0** |
| reported API cost（raw 10＋再走） | $0.04007244 |

これは、VAGOが報告した最良Cloud baselineの0.8 killとほぼ同じscore帯に着地した。ただし
model（こちらはLlama 3.1 8B）、seed、world clockが違うため「再現値」や直接勝敗とは呼ばない。

また、構造化V4の平均4.0とのscore差は入力だけのablationではない。VAGO contractへ合わせるため
action spaceも一文字motorからaction名＋同時押しへ変わっている。ここから安全に言えることは次である。

1. VAGO型の長い入力でも、3-lane Cloud LLM controllerは止まらないworldで技術的に成立した。
2. 失敗原因は文法崩壊ではない。958回答すべてがactionへ変換できた。
3. それでも0.7 killなので、overlapだけでは入力理解と古いactionの問題を救えない。
4. 次の公平な分解には、同じbody/promptで`v/x/a`とASCII＋depthだけを入れ替える入力ablationが要る。

生ログ:

- [raw 10 episode](results/vago-cloud-text-openrouter-groq-10x-20260823.json) — SHA-256
  `62132e34bcc4817f2b338676281a47784eea231023584cb7abd6e8cd79d06857`
- [seed 14 valid-clock rerun](results/vago-cloud-text-openrouter-groq-seed14-rerun-20260823.json) — SHA-256
  `2365e54b71d07e07f15e1c796a0bc2d3383fb6eac19d6dde7649edc4083c0c63`

## Colab二T4 — 同じ公開promptをローカルGPUで読む

2026-08-24、二つのColab T4へLlama 3.1 8B Instructを4-bit NF4で一体ずつ置き、同じ
VAGO Cloud text contractをseed 7〜16で実行した。三台目はColabの同時session上限で借りられなかった。

| 指標 | OpenRouter / Groq 3 lane | Colab 2 T4・cache前 | Colab 2 T4・system cache後 |
| --- | ---: | ---: | ---: |
| kill / 平均 | 7 / 0.7 | 9 / **0.9** | 9 / **0.9** |
| completed decision | 958 | 161 | 281 |
| model / server compute | 360.6 ms API completion | 1,387.9 ms | **1,040.4 ms** |
| observation-to-result | 386.3 ms | 1,546.2 ms | **1,193.0 ms** |
| action age | 11.75 tic | 53.89 tic | **41.39 tic** |
| mean native clock | 34.41 Hz | 34.99 Hz | 34.86 Hz |
| parser fallback / request error | 0 | 0 | 0 |

Colab版のcompletionは平均5.1 tokenしかない。それでもcache前は約1.39秒かかったため、長い回答では
なく、毎回平均697.7 prompt tokenをNF4 T4でprefillする時間が支配していた。Groqはこの長入力を
約4倍速く処理しており、**ローカルGPUだからCloudより低遅延、とは限らない**。

構造化V4と同じ発明を移植し、公開system promptのKVだけをlaneごとに保持した。ASCII画面、depth、
prompt本文、temperature 0.7、action parserは変更していない。10本比較ではcompute -25.0%、
delivery -22.8%、action age -23.2%、判断数 +74.5%まで改善したが、killは9のままだった。
速度改善は生存時間を12.35秒から17.02秒へ伸ばしたが、VAGO形式でのpolicy品質と長い可変suffixが
残り、score改善には直結しなかった。

生ログ:

- [Colab 2 T4 / cache前](results/vago-cloud-text-colab-2t4-uncached-10x-20260824.json) —
  SHA-256 `df1ceb115e07dc57da9111a26d8f148f6a78403db1794fafd271aa19b198e4d7`
- [Colab 2 T4 / system-prefix cache後](results/vago-cloud-text-colab-2t4-prefix-cache-10x-20260824.json) —
  SHA-256 `7b7a6dc8389ce004ffc13dd42d25b4dae48afab915593c3e0f5fcf123bbe34a2`
