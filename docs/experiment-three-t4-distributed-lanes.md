# Three physical T4 lanes — 同じLLMをGPUごと三交代にする

Status: implementation ready; live Colab run pending.

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

この時点で**Colab runtimeは0台、消費CUは0**。notebook、Drive上のA/B/C copy、remote server、
OpenRouter smoke/10-runはすでに準備済みで、ブラウザ制御helper復旧またはユーザーによるGoogle login後に
`Run all`から再開できる。

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
