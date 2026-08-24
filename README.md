# Latency Kills

> **「Cloud LLMはFPSには遅すぎる」を、モデルの賢さではなく時計とaction ageに分解して確かめる。**

[![tests](https://github.com/RPG-478/latency-kills/actions/workflows/tests.yml/badge.svg)](https://github.com/RPG-478/latency-kills/actions/workflows/tests.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Latency Killsは、Cloud LLMと小型専用modelを、35 Hzで止まらず動くoffline ViZDoomへ接続する
リアルタイム制御実験です。出発点は「LLMのthinkingをそのまま筋肉にしたら面白いのでは？」でしたが、
現在の主題は**actionが何tick古いとFPS agentは壊れるのか**です。

旧名は`Thought Leak Range`。raw thinkingを射撃へ漏らしたV0の技術名と、互換用Python distribution / CLI
`thought-leak-range`、module `thought_leak_range`にその痕跡を残しています。

オンラインゲームbotではありません。OSのキーボードやマウス、commercial game、anti-cheatには
接続せず、ViZDoomのローカル練習scenarioだけをAPI経由で操作します。

## 出発点: VAGOの1.3M DOOM agent

このrepositoryが追試する出発点は、VAGO Solutionsらの2026年のarXiv preprint
[Playing DOOM with 1.3M Parameters: Specialized Small Models vs Large Language Models for Real-Time Game Control](https://arxiv.org/abs/2604.07385)
です。著者はDavid Golchinfar、Daryoush Vaziri、Alexander Marquardt。
公開modelは[SauerkrautLM-Doom-MultiVec-1.3M](https://huggingface.co/VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M)、
実装は[GitHub](https://github.com/VAGOsolutions/SauerkrautLM-Doom-MultiVec)にあります。

この研究は、ViZDoomの`defend_the_center`を次の方法で解きます。

- 画面を40×25のASCIIへ変換し、16段階のdepth情報を加える
- 31,645 frameのhuman demonstrationで、4 actionを選ぶ専用classifierを学習する
- 5-layer ModernBERT-Hash＋attention poolingで、総parameter数を約1.3Mに抑える
- `shoot` / `move_forward` / `turn_left` / `turn_right`をmulti-labelで選ぶ

原論文の主結果は次の通りです。これは本repositoryの測定値ではなく、VAGO論文Table 2の報告値です。

| VAGO論文のagent | parameters | episodes | latency / decision | total frags |
|---|---:|---:|---:|---:|
| **SauerkrautLM-Doom-MultiVec** | **1.3M** | 10 | **31 ms** | **178** |
| GPT-4o-mini | proprietary | 10 | 646 ms | 0 |
| Gemini Flash Lite | proprietary | 10 | 920 ms | 8 |
| Nemotron-120B | 120B | 5 | 8.9 s | 3 |
| Qwen3.5-27B | 27B | 3 | 13.3 s | 2 |

*表1 — VAGO原論文Table 2の報告値。episode数がmodelごとに異なるため、total fragsだけを横並びにして
「同条件の平均score」とは扱わない。*

### このREADMEで使う言葉

| 言葉 | ここでの意味 |
|---|---|
| episode | ゲーム開始から死亡または制限時間までの1試行 |
| frag / kill | score上で敵1体分を倒したこと。monster同士の攻撃が加算される場合もあるため、実験ではhitやammoも併記 |
| game tic | ViZDoom内部時間の1刻み。35 Hz設定では約28.6 ms |
| action age | 画面を観測してから、その観測に基づく操作が実行されるまでの古さ |
| stopped / V4-S | Cloud返答待ちの間、ViZDoom内部時間も敵も停止する診断条件 |
| unpaused / 非同期 | modelの推論中もViZDoomが35 Hzで進み続ける条件 |
| formal run | score集計用の正式測定。録画負荷などで時計を乱さない |
| visual-only run | 挙動を目で確認するGIF用run。正式scoreには混ぜない |

つまり原論文の結論は明快です。**リアルタイム制御では、巨大な汎用LLMより、速くて小さい専用modelが
圧倒的に強い。** 本repositoryもこの勝敗自体には反論しません。実際、1.3M modelを止まらない世界へ
載せ直して、論文の17.8 frags/episodeに対して17.7を再現しました。

## では、何を調べ直したのか

VAGOの公開benchmark loopは、観測後にmodel/APIの返答を待ち、その後`make_action()`でViZDoomを
進める同期構造です。Cloud APIを待っているwall-clock時間中、game tickは進みません。これは実装を読んだ
だけでなく、同じpolicyへ0 ms / 650 msの待ち時間を入れ、RGB・depth・step・kill・HP trajectoryが
bit-identicalになることでも確認しました。詳細は[VAGO timing probe](docs/vago-sync-probe.md)にあります。

これはVAGOの1.3M modelが弱いという話ではありません。31 msなら世界を止めなくても十分に速いはずです。
一方、Cloud LLMの646 ms〜13.3 sというlatencyを「real-time game controlの失敗原因」として読むなら、
API待機中も世界が進む条件と、古いactionが身体へ届く条件を分けて測る必要があります。

そこで本repositoryでは、次の三つを作りました。

1. ViZDoomをmodel推論と独立した35 Hz threadで動かす、止まらないVAGO adapter
2. 一つのCloud LLMが一文字でWAIT / LEFT / RIGHT / FIREを直接選ぶCloud V4
3. 同じVAGO 1.3Mのaction到着だけを200 msへ遅らせるlatency ablation

調べたいのは「どちらのmodelが賢いか」ではなく、**同じ判断が何native tic古くなった時に使い物に
ならなくなるか**です。

## 今回得られた結果

同じseed 7〜16で、時計とaction latencyを明示して比較しました。

| 条件 | 推論中のworld | 観測からactionまで | kill合計 / 平均 |
|---|---|---:|---:|
| Cloud V4-S / Llama 3.1 8B | **停止** | 214.2 msのwall time | **263 / 26.3** |
| Cloud V4 / 同じLLM | 35 Hz継続 | 232.8 ms、約8 tic | **40 / 4.0** |
| VAGO MultiVec 1.3M / T4 | 35 Hz継続 | 28.1 ms、1.049 tic | **177 / 17.7** |
| 同じVAGO 1.3M + 200 ms floor | 35 Hz継続 | 7.038 tic | **42 / 4.2** |

*表2 — seed 7〜16の10 episode比較。停止Cloudの26.3は診断上限、残り3条件はworldが進み続ける。
Cloud V4とVAGOは入力・学習・action空間が違うため、4.0対17.7をmodel知能の直接対決とは扱わない。*

最初の行はworldを止めるdiagnostic上限であり、リアルタイムscoreではありません。1.3M modelは、
止まらない世界でもVAGO公開値17.8に対して17.7を再現しました。ところがmodel、weights、入力、
action policyを変えず、action到着だけを最低200 msへ遅らせると17.7から4.2へ76.3%低下。
別設計のCloud V4が出した4.0とほぼ同じscore帯へ着地しました。

> **1.3Mの専用脳から賢さを奪わず、反射神経だけCloud並みにしたら、戦績までCloud並みになった。**

これは「両modelの知能が同じ」、あるいは「Cloud V4がVAGOに勝った」という意味ではありません。
むしろVAGOの勝利を再現した上で、勝敗を説明する主要因の一つとしてaction stalenessを切り出した結果です。
現在の仮説は、リアルタイムAgentの能力をparameter数や平均推論msだけでなく、
**何native tic前の観測を操作しているか**でも測るべき、というものです。

- [四条件の詳細](docs/experiment-v4-vago-three-way.md)
- [200 ms latency ablation](docs/experiment-vago-1.3m-200ms-latency.md)
- [論文計画: Action Staleness](docs/paper-plan-action-staleness.md)
- [raw result JSON](docs/results/)
- [VAGO原論文と他の先行研究](docs/prior-art.md)

## 追試: 意味を一tokenへ戻し、古い命令だけ捨てる

上の主結果とは別に、Colab T4へLlama 3.1 8Bを常駐させ、V4の「速さ」と「意味」を分解した。
旧V4の数字`0`〜`5`は速い一方、未見座標の静的試験で29 / 53しか合わなかった。

`WAIT / LEFT_SHORT / ...`をそのまま生成させると48 / 53まで直ったが、平均1.984 tokenとなり
server computeは172.7 msへ悪化。そこで六操作を、全てtokenizer上1 tokenの
`wait / left / west / right / east / fire`へ符号化した。`west / east`はLONG旋回の筋肉コードで、
local側は返答を固定変換するだけで敵座標から操作を選び直さない。

| unpaused Colab条件 | kill平均 | 静的rule正答率 | 判断mean | server compute |
|---|---:|---:|---:|---:|
| 旧digit / 2 T4 / TTL 400 ms | **4.8** | 53.53% | 264.8 ms | 105.3 ms |
| semantic label / 2 T4 / TTL 400 ms | 3.4 | 84.54% | 335.6 ms | 172.7 ms |
| semantic one-token / 2 T4 / TTL 400 ms | 3.3 | 81.63% | 278.5 ms | 110.0 ms |
| semantic one-token / 1 T4 / TTL 400 ms | 4.1 | **85.34%** | 282.7 ms | 116.0 ms |
| **semantic one-token / 2 T4 / TTL 300 ms** | **4.6** | 83.69% | 273.3 ms | 110.1 ms |

全てseed 7〜16、35 Hz、40 ms観測、Flat-4。意味正答率を直してもscoreは単調に上がらず、二laneも
一laneより弱くなった。約0.28秒古い命令では、正確な現在座標追従や高い返答頻度が、そのまま
stale command競合になるためである。

action age上限を300 msへ縮めると、同じ二T4 V5が33→46 kill、平均3.3→4.6へ回復した。一方
250 msでは返答の意味正答率92.86%でも約8割を期限切れで捨て、平均1.4へ崩れた。今回の新しい結論は、
**命令は古すぎても駄目だが、新鮮さを求めすぎて身体を飢えさせても駄目**という制御帯域の崖である。
ただしTTL 300−400 msのpaired bootstrap 95% intervalは`[-0.4, 3.1]`で、10 seedではまだ
統計的に決着していない。4.6は現在の探索上の最良値であり、確定したSOTAや普遍的な最適TTLとは呼ばない。
さらに一台だけを300 msへすると、同じ5 seedで5.0→3.4へ低下した。二laneで返答供給を増やし、
TTLで古いtailを落とす二つを組み合わせた時だけ改善したため、帯域と鮮度は別々に最適化できない。

- [One-token semantic motorsの全比較](docs/experiment-semantic-one-token-motors.md)
- [V5 / 2 T4 / TTL 300 msの10本](docs/results/colab-t4-semantic-words-v5-nf4-ttl300-10x-20260824.json)

## 動いているところ

V4では、一つの汎用Cloud LLMがWAIT / LEFT / RIGHT / FIREとpulse長を一文字で直接選びます。
以下は、Marine認識修正後に取得した、worldを止めないFlat-4版のseed 08です。7 killしましたが、
GIF録画負荷がgame clockへ影響した**visual-only run**なので、上表のformal scoreには使っていません。

![止まらないCloud V4 Flat-4のvisual-only replay](https://github.com/RPG-478/latency-kills/releases/download/replays-v4-async-flat4-2026-08-21/seed-08__kills-07__health-neg02__seconds-14p7__visual-only.gif)

*GIF 1 — Cloud V4 / seed 08 / 7 kill / worldは停止しない。録画負荷でclockが乱れたvisual-only映像であり、
formal平均4.0の計算には含めていない。画面上の操作字幕はCloud LLMが返した一文字を固定変換したもの。*

### 途中で「チェーンソー男がLLMへ渡っていない」事故があった

以前ここに掲載していたV4 / V4-SのGIFは、現在版の代表映像として不適切でした。Freedoomの
チェーンソー兵は`MarineChainsawVzd`というactorですが、古い観測filterがその名前をmonsterとして
認識せず、座標をLLMへ渡していませんでした。LLMは見えている敵を無視したのではなく、prompt上では
その敵が存在していなかった、というsensor bugです。

修正では`MarineChainsawVzd`の互換名とViZDoomのMonster categoryを認識し、同時に一体だけを見る
target lockを追加しました。修正直後のV4-Sをseed 7〜24で18 episode走らせると、215 kill、平均11.94。
ログ上でもMarineを観測し、少なくとも146 killをMarineへのFIREへ帰属できました。

次は修正済みV4-S seed 10の14-kill replayです。これはCloud待機中にworldを止めるcorrectness診断で、
リアルタイム性能ではありません。

![MarineChainsawVzd認識修正後のV4-S seed 10](https://github.com/RPG-478/latency-kills/releases/download/replays-v4-s-marine-2026-08-21/seed-10__kills-14__hits-14__health-100__ticks-525__complete.gif)

*GIF 2 — Marine認識修正後のV4-S / seed 10 / 14 kill・14 hit / 525 tic完走。Cloud待機中は敵も時計も
停止するため、これはLLM＋Systemの対応を調べる映像であり、リアルタイム性能の証拠ではない。*

ただし、修正後も「LLM＋Systemが完全に正しい」とは判定できませんでした。Marineが照準より少し右に
いる段階でFIREを繰り返し、敵が横移動または接近して当たり判定へ入った時に倒す場面が多く、
Marineへの実弾命中率は39.7%でした。つまり認識漏れは直った一方、早すぎるFIREと照準誤差は別問題として
残っています。詳細と18本すべてのGIFは
[Marine recognition repair baseline](docs/replays/2026-08-21-v4-s-marine-fixed-before-overshoot/README.md)にあります。

[全replayとformal / visual-onlyの区別](docs/replays/README.md)も公開しています。GIFはrepository本体を
肥大化させないよう、実験別GitHub Releasesへ置いています。

## 仕組み

### Cloud V4

~~~text
ViZDoom labels: 最接近敵の v / x / ammo
                  │  0.1秒ごと、3 laneを時間差で重ねる
                  ▼
      Llama 3.1 8B via OpenRouter / Groq
                  │  streamed first valid token
                  ▼
   0 WAIT / 1,2 LEFT / 3,4 RIGHT / 5 FIRE
                  │
        文法・観測番号・TTLだけ検査
                  ▼
     35 Hzで止まらないViZDoomへ直接入力
~~~

local側は敵位置から操作を選び直しません。LLMの誤答も、期限と文法を満たせばそのまま実行します。
3 laneは三人格の会議ではなく、同じpolicyを重ねてnetwork待ちを隠す非同期pipelineです。

### VAGO 1.3M adapter

~~~text
screen + depth ──> upstream ASCII/depth converter ──> 1.3M policy on T4
       ▲                                                   │
       │  latest-only observation                          ▼
35 Hz PLAYER thread <── 4-tic composite action <── process boundary
~~~

ViZDoomを専用threadが35 Hzで進め、model processが遅れてもworldを止めません。古い未処理frameは
latest-onlyで置換し、前actionは4 ticで必ず失効します。200 ms実験ではmodel計算後に
「観測時刻＋200 ms」まで待ち、遅い推論のlatencyとthroughputを模擬します。

## 何がLLMで、何がlocalか

| mode | LLMが決めること | local側 |
|---|---|---|
| fire-gate | 数秒のFIRE許可 | 探索、追尾、中央確認、実射 |
| direct-bit | 一発ごとのFIRE / WAIT | 左右追尾 |
| four-agent / V3 | 4 AgentがWAIT / LEFT / RIGHT / FIREへ投票 | 期限付きarbitration |
| direct-motor / V4 | **全操作とpulse長** | 一文字の固定変換と安全検査 |
| VAGO adapter | upstream 1.3M policyのcomposite action | clock、queue、action expiry |

「Cloud LLMがFPSを全部操作した」と呼ぶ主対象はV4です。fire-gateはLLM＋local spineの
階層制御実験であり、LLM自身が照準を合わせた結果とは数えません。

## 再現

### 1. 課金も通信もないmock

Python 3.12と[uv](https://docs.astral.sh/uv/)を使います。

~~~powershell
git clone https://github.com/RPG-478/latency-kills.git
cd latency-kills
uv sync --extra dev --locked --python 3.12
uv run python -m thought_leak_range mock --duration 6
~~~

### 2. Cloud V4

OpenRouter keyはprocess環境またはignored .envから読みます。repository、引数、ログには書かないでください。

~~~powershell
$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API key"
uv run python -m thought_leak_range live --model meta-llama/llama-3.1-8b-instruct --provider Groq --no-provider-fallback --tap-mode direct-motor --lanes 3 --scenario defend_the_center --duration 15 --observation-interval 0.10 --motor-token-max-age-ms 400 --world-clock clock-thread --motor-body clock-thread --max-tokens 16 --max-requests 180 --max-usd 0.025
Remove-Item Env:OPENROUTER_API_KEY
~~~

model/providerの提供状況と価格は変わり得ます。必ず小さい--max-usdと--max-requestsから始めてください。

### 3. 止まらないVAGO 1.3M

VAGO repositoryとcheckpointは外部入力です。このrepositoryにはweightsもupstream policyも同梱しません。

~~~powershell
uv run python tests/manual_vago_multivec_async_batch.py --upstream C:\path\to\SauerkrautLM-Doom-MultiVec --scenario-path C:\ascii-path\defend_the_center.cfg --device cuda --seeds 7 8 9 10 11 12 13 14 15 16 --output runs\vago-1.3m-async.json
~~~

200 ms floorを再現するには--minimum-action-latency-ms 200を追加します。episodeごとのOS process隔離と
watchdogは、ColabでViZDoomが三episode目に停止した再現性事故を黙って平均から外さないためのものです。

## 実験環境

### このPC

| 項目 | 値 |
|---|---|
| OS | Windows 11 Home 64-bit、10.0.26200 build 26200 |
| CPU | 12th Gen Intel Core i5-1235U、10 cores / 12 logical processors |
| GPU | Intel Iris Xe Graphics、driver 32.0.101.5542、shared integrated memory |
| RAM | 7.7 GiB usable |
| experiment Python | CPython 3.12.10 via uv |
| system Python | CPython 3.14.2（project対象外） |
| uv / Git / PowerShell | uv 0.11.31 / Git 2.52.0.windows.1 / PowerShell 7.6.4 |
| project runtime | httpx 0.28.1 / Pillow 12.3.0 / ViZDoom 1.3.0 |

このPCでVAGO公式同期runnerをCPU再現した時は、Torch CPU推論が平均216.2 msでもworldが停止するため
平均15.6 killでした。GPUを持たないこのPCで独立35 Hz clockとCPU推論を同時に走らせると、model processが
game threadまで飢えさせ14〜20 Hzへ落ちました。したがって正式な止まらない1.3M値はColab T4で取得しています。

### 外部実行環境

| 用途 | 環境 | 主な実測 |
|---|---|---|
| Cloud V4 | OpenRouter、Groq、Llama 3.1 8B Instruct | mean 232.8 ms、平均4.0 kill |
| Remote semantic V5 | Google Colab T4 × 2、Quick Tunnel、Llama 3.1 8B NF4 | compute 110.1 ms、wire 245.9 ms、TTL 300 msで平均4.6 kill |
| 1.3M async | Google Colab、Tesla T4、Python 3.12 | compute 28.1 ms、35.019 Hz、平均17.7 kill |
| 1.3M + 200 ms | 同じColab T4 | action age 7.038 tic、35.045 Hz、平均4.2 kill |

公開する環境情報からは、account、メール、端末名、ユーザー名、serial、IP、API key、Colab認証情報を除外しています。

## 公平性と限界

- Cloud V4はViZDoom labelsから最接近敵一体の構造化位置を読む
- VAGOはscreenを40×25 ASCIIへ変換し、16段階depthも読む
- Cloud V4は一文字につき一種類、VAGOは前進・旋回・射撃を同時押しできる
- VAGOは31K human-play framesで学習したDoom専用model、V4は汎用Cloud LLM
- V4-Sの26.3は推論中にworldを止めた診断値で、リアルタイム勝利とは呼ばない
- GIF captureはnative clockを遅くし得るため、formal数値runとvisual-only runを分ける
- KILLCOUNTにはmonster infightingが入り得るため、hit、ammo、damageも併記する
- seed 7〜16の10本は強い予備結果だが、一般化にはseed増量と複数scenarioが必要

このprojectは世界初を主張しません。Cloud LLMをViZDoomへ接続した先行例はあります。差分と、
どこまで反例になるかは[prior artと公平性](docs/prior-art.md)へ明記しています。

## 実験史

| 世代 | 発明と失敗 | 主な結果 |
|---|---|---:|
| V0 | raw reasoning中の自然言語を直接motorへ漏らす | 固定標的へ実射 |
| V2 | LLMは引き金、35 Hz local spineが追尾 | 10 run平均5.4 |
| V3 | WAIT / LEFT / RIGHT / FIREの4 Cloud Agent | 平均1.5、会議が身体を止めた |
| V4 | 一つのLLMが一文字で全操作 | 旧15秒run平均3.5 |
| V4-S | 同じV4で推論中のworldを停止 | flat-4平均26.3 |
| V4 Async | worldとmotorを独立35 Hz clockへ修理 | flat-4平均4.0 |
| VAGO Async | 1.3M専用modelを同じ止まらない世界へ | 平均17.7 |
| VAGO 200 ms | 1.3Mのaction鮮度だけCloud級へ落とす | 平均4.2 |
| Remote V5 | 意味を持つ六つの1-token motor word＋action TTL | TTL 300 msで平均4.6、250 msで1.4 |

途中の黒板干渉、敵label取りこぼし、LONG=20 tic、overshoot、CPU starvation、Doom lifecycle hangも
失敗ごと公開しています。[documentation map](docs/README.md)から全記録へ進めます。

## 研究の次

- 30 / 60 / 100 / 150 / 200 / 233 / 300 msのlatency cliffを描く
- 固定delayと同じ平均を持つCloud型jitterを比較する
- stale actionを実行 / 破棄 / 予測補正する条件を比べる
- semantic V5で見つかった250 / 300 / 350 / 400 msのTTL崖をseed・scenarioを増やして再検証する
- seed数とscenarioを増やし、bootstrap confidence intervalを出す
- action ageを第一級metricとして、robotics / VLA / networked controlの先行研究を再監査する

詳しくは[paper plan](docs/paper-plan-action-staleness.md)。再現、反証、もっと良い身体、面白い失敗のIssueを歓迎します。

## 生成物と安全境界

runは既定でruns/<timestamp>/へevent、summary、任意のGIF / raw thoughtを保存します。runs/、
.env、virtualenv、build output、GIFはGitから無視されます。

- output tokenはrequest、観測番号、TTLへ束縛し、期限切れ・逆順・重複を拒否
- actionはWAIT / LEFT / RIGHT / FIREのallow-list
- leaseが切れたtickは全ボタンを離す
- model出力をshell、Python、任意codeとして実行しない
- API keyはartifactへ保存せず、HTTP errorでもredact
- live modeはrequest数とUSD budgetを起動前から制限
- native keyboard / mouse inputを実装しない

脆弱性はpublic Issueへ秘密情報を貼らず、[Security policy](SECURITY.md)に従ってprivate advisoryへ報告してください。

## Test

~~~powershell
uv sync --extra dev --locked --python 3.12
uv run python -m pytest -q
~~~

CIはUbuntu / Windows、Python 3.12で実行します。

## Licenseとupstream

Latency Kills自身は[MIT License](LICENSE)です。

VAGOの[Hugging Face model card](https://huggingface.co/VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M)と
GitHub READMEはApache-2.0を表示しています。一方、2026-08-23のupstream commit `b4c3fdf`にも
`LICENSE` fileと`pyproject.toml`のlicense fieldはなく、GitHub API上のlicenseも未判定です。
本repositoryはupstreamのweights、source、action policy、promptを再配布せず、
ユーザーが別途用意したcheckoutを動的に読み込むadapterだけを提供します。sourceを利用する場合は
upstreamの最新表記を確認してください。

---

**くだらないことを、ちゃんと作る。**
