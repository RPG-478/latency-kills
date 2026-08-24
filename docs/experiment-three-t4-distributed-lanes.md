# Three physical T4 lanes — 同じLLMをGPUごと三交代にする

Status: **two physical T4 lanes measured on 2026-08-24**. The planned third
runtime was rejected by Colab's concurrent-session limit. A later semantic-policy
ablation found that two lanes recover from 3.3 to **4.6 kills/game** when stale
commands older than 300 ms are rejected.

## 2026-08-24 追試: 二台を強くしたのは三台目ではなくTTLだった

旧digit policyが未見座標で29 / 53まで落ちたため、入力を`DIRECTION + OFFSET`へ分け、
出力を一語の`wait / left / west / right / east / fire`へ変えた。六語は全てLlama 3.1 8Bで
1 tokenであり、local側は語を操作へ固定変換するだけで座標判断をしない。

このV5は53座標で47 / 53、server compute 109.4 msまで改善したが、TTL 400 msの二laneでは
平均3.3 kill。一laneの4.1より弱かった。throughputを増やす二台化が、時間差の古い旋回の競合まで
増やしたと考えられる。

そこでmodel、prompt、二T4、seedを固定し、action ageの上限だけを変えた。

| semantic V5条件 | runs | kill平均 | 返答 | accept率 | 静的rule正答率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 T4 / TTL 250 ms | 5 | 1.4 | 406 | 19.2% | 92.86% |
| **2 T4 / TTL 300 ms** | **10** | **4.6** | **975** | **70.9%** | **83.69%** |
| 2 T4 / TTL 350 ms | 5 | 3.6 | 487 | 87.7% | 83.57% |
| 2 T4 / TTL 400 ms | 10 | 3.3 | 909 | 91.9% | 81.63% |
| 1 T4 / TTL 400 ms | 10 | 4.1 | 464 | — | 85.34% |
| 1 T4 / TTL 300 ms | 5 | 3.4 | 217 | 70.0% | 92.63% |

250 msは返答の意味正答率が最も高いのに、約8割を期限切れとして捨てるため操作が飢えて最弱になった。
400 msは供給量が多いが古い命令も通す。今回の分布では中間の300 msが46 kill、平均4.6で、同じ
二lane / 400 msの33 killを13上回った。旧digit版の48 killには2届かない。
ただし300−400 msのpaired bootstrap 95% intervalは`[-0.4, 3.1]`、exact sign-flipは`p=0.25`。
10 seedの探索結果であり、優越が統計的に確定したとは扱わない。

一台だけを300 msへ縮めた追加5本は平均3.4で、一台400 msの同じ5 seed平均5.0から低下した。
一台300 msは152 actionしかacceptできず、二台300 msは334。今回の改善はTTL単独より、
**二laneの供給量と300 msの鮮度cutを組み合わせた時**に現れた。

詳しいpolicy比較、paired seed、V6の失敗、生ログは
[One-token semantic motors](experiment-semantic-one-token-motors.md)へ分離した。

## 2026-08-24 実測結果

有料ColabへA/B/Cの三notebookを用意したが、このaccountで同時に割り当てられたGPU runtimeは
二つまでだった。A/CへLlama 3.1 8B Instructを4-bit NF4で一体ずつ置き、Bは
`セッションが多すぎます`で未割当のままにした。したがって、これは三台実験の捏造版ではなく、
**一台対二台の実測**である。

共通条件は`defend_the_center`、seed 7〜16、unpaused `clock-thread`、Flat-4、TTL 400 ms、
local aim assistなし。数字六択のlogit制約も使っていない。

| 条件 | kill / 平均 | 判断数 | 判断mean / p50 / p95 | mean Hz | valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 T4 / 1 lane / 100 ms観測 | 39 / **3.9** | 529 | 294.2 / 297 / 375 ms | 33.05 | 10 / 10 |
| 2 T4 / 2 lane / 100 ms観測 | 44 / **4.4** | 1,061 | 263.7 / 250 / 344 ms | 33.03 | 10 / 10 |
| 2 T4 / 2 lane / 40 ms観測 | 48 / **4.8** | 1,076 | 264.8 / 265 / 343 ms | 33.12 | 10 / 10 |

二台化で判断数は`529 → 1,061`、**+100.6%**。物理GPU分散は意図どおり制御帯域をほぼ
完全に二倍へした。一方、killは`3.9 → 4.4`、+0.5に留まった。seedごとのpaired比較は
二台が6勝、1分、3敗である。観測を100 msから40 msへ縮めても判断数は`1,061 → 1,076`
しか増えず、killは4.4から4.8だった。二laneが埋まった後の新観測はcoalesceされるため、
観測生成だけを速くしても判断帯域は増えない。

| remote内訳 | 1 T4 | 2 T4 / 100 ms | 2 T4 / 40 ms |
| --- | ---: | ---: | ---: |
| server compute | 111.2 ms | 111.0 / 99.4 ms | 111.1 / 99.2 ms |
| public-tunnel wire | 232.7 ms | 235.4 / 243.9 ms | 227.0 / 249.6 ms |
| request error | 0 | 0 | 0 |

T4内部は約100〜111 msまで来たが、Cloudflare Quick Tunnelを含むwireは約227〜250 ms。
三台目が借りられても一判断の古さは消えず、増えるのは主にcadenceである。今回cadenceを二倍に
してもscoreが比例しなかったため、次のbottleneckはstale action、policy誤り、旋回overshoot、
射撃機会の位相にある。

raw CLI summary 30本は
[`colab-t4-structured-30x-20260824.json`](results/colab-t4-structured-30x-20260824.json)
（Git LF blob SHA-256 `bc72514794b4ba549dc0d354febaa0c1c746aaa8273ca0a47cb3b56cc318d65c`）。
runtime bearer tokenとendpointは含めていない。

### 採用しない最初の1本

最初はscenario指定を忘れて`basic`を実行し、1 kill時点の2.03秒でepisodeが終わった。通信確認には
使えたが、15秒の性能値には混ぜていない。この失敗で`--duration 15`だけではscenarioの終了条件を
上書きしないことを再確認した。

### 起動中に直した二つのバグ

1. 共通probe loggerがOpenRouter client固有の`.model`を仮定し、remote clientで落ちた。
   remote poolへ非秘密の識別名を追加して修正した。
2. Cloudflare URL発行直後、Colab自身のDNSに名前がまだ現れず、一回だけのpublic health checkが
   落ちた。最大120秒のbounded retryへ修正した。

### さらに見つかった本命: 6 / 6 probeは境界理解を保証していなかった

上の三条件はすべてgame開始前の六択probeを6 / 6で通った。しかし実戦ログの
`expected_token`と実出力を照合すると、意味正答率は一台100 msが56.33%、二台100 msが
46.37%、二台40 msが53.53%しかなかった。二台100 msでは1,061判断のうち610回がFIREで、
そのうち460回はrule上まだ左右旋回すべき位置だった。

原因を分けるため、各物理laneへ同じ座標を直接送るdecision-boundary sweepを追加した。
敵なし・弾なしに加えて`x=-500..500`を20刻みで53ケース測ると、両laneとも**29 / 53**。
しかも53 / 53で二台の答えが完全一致した。したがってT4個体差やtunnelの化けではなく、
同じmodelとpromptが同じように数値境界を誤読している。

| 観測 | 期待 | 実際 | 注記 |
| --- | --- | --- | --- |
| `x=-351` | LEFT_LONG | RIGHT_LONG | 境界から十分左でも左右反転 |
| `x=-350` | LEFT_LONG | LEFT_LONG | system prompt中の例そのもの |
| `x=-349` | LEFT_LONG | LEFT_SHORT | 1だけ動くと別class |
| `x=-81` | LEFT_SHORT | FIRE | まだ中央の外 |
| `x=-80` | FIRE | LEFT_SHORT | 境界点で逆転 |
| `x=149` | RIGHT_SHORT | FIRE | startup probeの1手前 |
| `x=150` | RIGHT_SHORT | RIGHT_SHORT | startup probe点だけ正解 |
| `x=151` | RIGHT_SHORT | FIRE | 1だけ動くと再び誤り |

従来probeの6ケース中4ケースはsystem prompt中の例そのもので、残る左右SHORTも
`-150 / 150`という固定の代表点だった。これは「六つのcanonical pointを返せる」試験であり、
連続座標の区間を理解した証明ではなかった。挙動は**例題・probe点へのanchoringと壊れやすい
数値補間に整合する**。内部機序を直接観測したわけではないので、単純な暗記と断定はしない。

この発見により、二台化で判断数だけ倍増した説明も変わる。増やしていたのは正しい判断だけではなく、
右にいる敵へ早すぎるFIREを返す判断も含む。`3.9 → 4.4 → 4.8`をGPU台数だけの限界と読む前に、
holdout座標を含むprobeとpolicyの修正が必要である。

生ログ:

- [20刻み53ケース・二lane](results/remote-motor-boundary-sweep-2t4-20260824.json) —
  Git LF blob SHA-256 `0fe1e541bfbd18942dae59552a3b2dd30dbb7fe245635247d559067fd71f3395`
- [境界・probe近傍29ケース・二lane](results/remote-motor-probe-neighborhood-2t4-20260824.json) —
  Git LF blob SHA-256 `a61a28b01c67d35b7eccfda8904b20ef44cb5e6b173d189062676f6e1cfb5bb2`

次版の起動試験はprompt例と同じ値を合格判定へ使わず、境界の両側、未見の区間内部、乱数seedを
固定したholdout sweepを別に採点する。現在の6 / 6は後方互換のsmoke testとして残すが、
「運転免許」ではなく配線確認へ格下げする。

## 2026-08-24 runtime起動前のbrowser failure

ユーザーからT4三台、Colab dependency導入、Colab Secretsの`HF_TOKEN`送信についてaction-time
confirmationを得た後、三notebookの起動を試みた。しかしruntime接続前にブラウザ制御層が停止した。

- in-app browserの永続Node sessionはreset後も`failed to write kernel assets (os error 3)`。
- ChatGPT app本体をWindows UI automationで触ることは禁止されているため、その迂回は不採用。
- 既存Chrome profileをPlaywrightで直接開くと接続が成立せず、起動したabout:blank processは
  開始時刻とPIDを照合して終了した。
- profileの一時copyではGoogleのdevice-bound sessionが移らずlogin画面になった。credential入力は
  自動化せず停止し、一時copy 591.3 MBはRecycle Binへ送り、元profileは変更していない。
- local debug付きChrome起動はsecurity policyで拒否されたため、回避しなかった。

この時点では**Colab runtimeは0台、消費CUは0**だった。その後、Codex内蔵browserの別経路で
復旧し、上記の二T4実測まで到達した。この失敗節は「GPUを借りる前に、GPUを借りる指が
bottleneckになった」記録として残す。

## 発端

一台のGoogle Colab T4へLlama 3.1 8B Instructを4-bit NF4で置き、固定system
promptのKV cacheを再利用すると、一つのV4判断は平均`107 ms`まで短縮できた。
ところが同じT4へ3 laneを重ねた既存実験では、GPU競合によって一判断が平均
`215 ms`まで悪化した。

そこで、同じpolicyを三つのT4へ一体ずつ複製し、各GPUを物理的な一laneとして使う。

```text
35 Hz ViZDoom
      │  newest structured observation
      ├── lane A ── T4 A / Llama 3.1 8B / cached prefix
      ├── lane B ── T4 B / Llama 3.1 8B / cached prefix
      └── lane C ── T4 C / Llama 3.1 8B / cached prefix
                         │
              first one-character motor token
```

三人格の会議ではない。三体とも同じmodel、同じprompt、同じV4 policyであり、時間差の
観測を独立に処理する。

## いちばん大事な区別

GPUを三台にしても、一判断そのものが`107 / 3 ms`になるわけではない。

- **decision latency / action age**: 一つの観測が一文字になるまで。およそ
  `107 ms + network RTT`のまま。
- **decision cadence / throughput**: 新しい一文字が届く間隔。三laneならGPU競合なしに
  高頻度化できる可能性がある。

したがって本実験は、「三台なら判断が三倍速い」ではなく、**古さは残したまま更新頻度
だけを上げると、FPSの成績は改善するか**を測る。

## 回収済みの既存値

2026-08-20〜21のColab notebookから再確認した値。scenarioは注記がない限り
`defend_the_center`、入力はV4の構造化`v / x / a`である。

| 条件 | 一判断 | 10 episode |
| --- | ---: | ---: |
| 一台T4・一lane・cached prefix | 平均107 ms | 平均2.4 kill |
| 一台T4・三lane・同一GPUで競合 | 平均215 ms | 平均4.0 kill |
| OpenRouter / Groq・Cloud V4 | 平均232.8 ms | 平均4.0 kill |

一laneは遅延が短いのに平均killが低い。これは当時の観測間隔が`200 ms`で、更新頻度が
低かったことも混ざる。三T4実験では観測間隔とlane数を独立に振る。

## 実装

- [`colab/remote_t4_lane.ipynb`](../colab/remote_t4_lane.ipynb): 一つのT4を一つの
  remote laneにするnotebook
- [`colab/remote_lane_server.py`](../colab/remote_lane_server.py): 固定prefix KV cacheと
  一文字motor endpoint
- `remote-live`: 一endpointにつき一つの永続HTTP connectionを割り当てるCLI
- bearer tokenはruntimeごとに生成し、command lineやGitHubへ置かない
- notebook outputはprivateにし、実験終了時は`Disconnect and delete runtime`で三台とも返す

public tunnelの往復を含む`wire_ms`と、T4内部の`server_compute_ms`を別々に記録する。

## 最初の実験行列

同じseed 7〜16、同じFlat-4 action、同じ35 Hz clockを基本とする。

| 条件 | 物理GPU | lane | 観測間隔 | 分けたい効果 |
| --- | ---: | ---: | ---: | --- |
| A | 1 | 1 | 100 ms | remote tunnel込みの単一lane基準 |
| B | 3 | 3 | 100 ms | 同じ要求率でGPU競合を除去 |
| C | 3 | 3 | 40 ms | cadenceを35 Hzへ近づける |
| D | OpenRouter | 3 | 100 ms | 既存Cloud V4の再確認 |

追加で、server側の「数字六択へのlogit制約あり／なし」を分ける。制約なしを本線とし、
制約ありは文法エラーだけを消すablationとして扱う。

## 成功・失敗どちらでも面白い点

- `server_compute_ms≈107`なのに`wire_ms≈230`なら、三T4はCloudflare往復に食われる。
- action ageが同じでもcadenceだけでkillが伸びれば、「古さ」だけでなく「制御帯域」が
  独立の説明変数になる。
- 三台にしても伸びなければ、V4のovershootとstale directionがボトルネックである。
- 一台T4のcontinuous batchingや、より速いquantized runtimeが三台を上回る可能性もある。

どの結果でも、巨大モデル対小型モデルという比較から、**知能・遅延・更新頻度・入力難度を
別々に測る**方向へ研究を一段進められる。
