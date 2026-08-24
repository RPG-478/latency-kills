# 論文計画: FPS agentを壊すのは推論時間か、actionの古さか

## 仮タイトル

**How Many Ticks Old Are You? Action Staleness as a First-Class Variable in Real-Time Game Agents**

日本語なら、**「何tick前の自分を操作している？リアルタイムゲームAgentにおけるAction Staleness」**。

## 中心仮説

リアルタイムFPS agentの性能は、model sizeや平均推論時間だけでは説明できない。
観測からactionが実行されるまでにworldが進んだ量、すなわち`action age`が独立した主要因である。

今回の予備結果では、同じVAGO 1.3M policyを変更せずaction到着だけを約1.05 ticから
7.04 ticへ遅らせると、平均killが17.7から4.2へ76.3%低下した。別系統のCloud V4は
約8 ticで平均4.0だった。

## 論文にできる貢献候補

1. **停止世界benchmarkの問題を定量化する**
   - 推論中にsimulationを止める評価と、35 Hzを維持する評価を分離する。
2. **modelを固定したlatency介入を行う**
   - 賢さ、weights、入力、action policyを変えず、到着時刻だけを操作する。
3. **wall-clock latencyでなくaction ageを第一級metricにする**
   - msだけでなく、観測後に何native tic進んだ命令かを測る。
4. **latency cliffを測る**
   - 30〜300 msでscoreが滑らかに落ちるか、特定tickで崖になるかを調べる。
5. **staleness対策を同じ物差しで比較する**
   - latest-only、TTL、破棄、予測補正、action chunk、world停止を比較する。
6. **鮮度だけでなくaction supplyも測る**
   - TTLを短くすると古い命令は減るが、身体へ届く操作も減る。accept率とneutral ticを
     action ageと同時に報告し、「新鮮だが飢えたcontroller」を分ける。

## 2026-08-24 TTL予備実験

Llama 3.1 8B、semantic one-token policy、二T4、35 Hz clockを固定し、action age上限を
250 / 300 / 350 / 400 msへ振った。seed 7〜11の平均killは`1.4 / 4.4 / 3.6 / 3.6`、
accept率は`19.2 / 69.2 / 87.7 / 90.6%`だった。250 msでは返答の静的意味正答率が
92.86%まで上がったのに、操作が飢えて最弱になった。

300 msを10 seedへ延長すると平均4.6、同じV5 / 400 msは3.3。ただしpaired bootstrap
95% intervalは`[-0.4, 3.1]`、exact sign-flipは`p=0.25`で、優越は未確定である。
この予備結果はTTL曲線が単調でない可能性と、accepted-action bandwidthを共変量でなく
第一級metricにする必要を示す。[完全な記録](experiment-semantic-one-token-motors.md)。

追加の5 seed 2×2では、一laneのTTL 400→300 msは平均5.0→3.4、accept 232→152へ低下。
二laneでは400→300 msが3.6→4.4、accept 396→334だった。厳しいTTLを単独で使うと身体が飢え、
複数laneの供給量と組み合わせた時だけstale tailを切れる、というinteraction仮説が生まれた。

## 最低限必要な本実験

- latency floor: 30 / 60 / 100 / 150 / 200 / 233 / 300 ms
- 各条件を十分なseed数で実行し、平均・中央値・bootstrap confidence intervalを出す
- 固定delayと、同じ平均を持つCloud型jitterを比較する
- action age、kill、survival、neutral tic、観測置換、p95 latencyを保存する
- TTL sweepは条件順をrandomize / interleaveし、accepted・expired・committed action率を保存する
- pulse長、observation interval、queue方式を個別ablationする
- 同一model内のpaired comparisonを主結果にし、Cloudとの比較は補助結果にする
- 少なくとも複数scenarioで再現し、`defend_the_center`固有現象でないことを確認する
- 実時間clockが全episodeで許容範囲内かを事前登録した基準で判定する

## 強く言えること／まだ言えないこと

### 現在でも強く言える

- VAGO 1.3Mは止まらない35 Hz世界でも平均17.7を維持した
- 同一policyへ200 ms floorを入れるだけで平均4.2まで低下した
- clock低下やmodel変更を伴わず、action ageは約6 tic増えた
- semantic V5の250 ms TTLは返答正答率が高くてもaction supply不足で平均1.4まで崩れた

### 本実験前には言わない

- すべてのFPS agentで同じ閾値になる
- Cloud LLMとVAGOの知能が同等である
- latencyだけが唯一の性能要因である
- TTL 300 msが普遍的な最適値である
- action ageという概念自体が新規である

最後の新規性は、robotics、real-time RL、networked control、game agent、VLAの先行研究を
改めて系統検索してから決める。新規性が概念そのものになければ、**FPSでの制御された実証、
停止benchmarkとの三者比較、latency cliffの測定法**を貢献の中心に置く。

## 一枚で見せる主図

横軸をaction age（tic）、縦軸をkillまたは正規化scoreにする。各latency条件を点で置き、
固定delayとjitterを別線にする。停止世界は`age = 0`ではなく、simulation clockを止めた別条件として
図の外側または破線で示す。

この図で「31 msの1.3M」と「233 msのCloud」をmodel名ではなく時間軸へ並べられれば、論文の顔になる。
