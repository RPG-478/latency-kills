# One-token semantic motors — 意味を直したら弱くなった

実験日: 2026-08-24

Status: **V5を採用、V6の手書きleadは棄却。2 T4 + TTL 300 msが現在の最良semantic条件。**

## 一言でいうと

数字`0`〜`5`を答えさせた旧V4は速いが、未見座標で`29 / 53`しか意味が合わなかった。そこで
LLMへ`WAIT`や`RIGHT_LONG`をそのまま書かせると意味正答率は上がったが、ほぼ2 tokenになって
遅くなり、killも落ちた。

最後に、意味を保ったまま全操作を1 tokenへ収めるため、次の妙な運動語彙を作った。

| LLMの出力 | 実行する操作 |
| --- | --- |
| `wait` | WAIT |
| `left` | LEFT_SHORT |
| `west` | LEFT_LONG |
| `right` | RIGHT_SHORT |
| `east` | RIGHT_LONG |
| `fire` | FIRE |

Llama 3.1 8Bのtokenizerでは六語すべてが1 tokenだった。`west` / `east`は地図の方角ではなく、
**旋回強度を一語へ押し込むための筋肉コード**である。local側は返った語を固定表で数字へ変換するだけで、
敵座標を見て操作を選び直さない。

## 入力も数字比較から一段だけ意味へ寄せた

旧入力は`v=1 x=-137 a=10`だった。V5では同じ最接近敵一体の情報を、server側で次の形へ直す。

```text
TARGET=VISIBLE DIRECTION=LEFT OFFSET=137 AMMO=10
```

符号付き座標を左右と絶対距離へ分けただけで、照準、射撃閾値、SHORT / LONG、探索は依然として
LLMが選ぶ。prefix KV cacheを保持し、観測suffixから最初のmotor wordだけを生成する。

## 静止した座標問題では大幅に改善した

| policy | 出力 | 20刻み53問 | 境界近傍29問 | server compute | wire |
| --- | --- | ---: | ---: | ---: | ---: |
| 旧digit V4 | `0`〜`5` | 29 / 53 | — | 約105 ms | 約237 ms |
| semantic-action V4 | `LEFT_SHORT`等 | **48 / 53** | **26 / 29** | 172.0 ms | 323.7 ms |
| semantic-words V5 | `left` / `west`等 | 47 / 53 | 25 / 29 | **109.4 ms** | **274.2 ms** |

長いaction labelは意味を直したが、10 episodeでcompletionは平均1.984 token、computeは172.7 msまで
増えた。V5はほぼ同じ座標正答率を1 token・約109 msへ戻した。つまり、**意味のある出力と一文字級の
反射速度は両立できた**。

ただしV5にも、`±240 / 260 / 280`をLONGでなくSHORTにする帯が残る。29問の境界近傍では
`±81`をFIRE、`±221`をSHORTにする誤りもあった。6 / 6のstartup probeだけを運転免許とは呼ばない。

## ところが、ゲームでは「正しいほど強い」にならなかった

共通条件はseed 7〜16、`defend_the_center`、35 Hz `clock-thread`、40 ms観測、Flat-4、
local aim assistなし。TTL 300 ms以外は400 msである。

| 条件 | kill合計 / 平均 | 返答数 | 静的rule正答率 | 早すぎるFIRE | 判断mean | compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| digit / 2 T4 | **48 / 4.8** | 1,076 | 53.53% | 394 | 264.8 ms | 105.3 ms |
| action label / 2 T4 | 34 / 3.4 | 731 | 84.54% | 83 | 335.6 ms | 172.7 ms |
| V5 / 2 T4 / TTL 400 | 33 / 3.3 | 909 | 81.63% | 46 | 278.5 ms | 110.0 ms |
| V5 / 1 T4 / TTL 400 | 41 / 4.1 | 464 | **85.34%** | 25 | 282.7 ms | 116.0 ms |
| **V5 / 2 T4 / TTL 300** | **46 / 4.6** | 975 | 83.69% | 75 | 273.3 ms | 110.1 ms |

数字版は静的rule上の誤答を394回もFIREへ変えたのに最強だった。これは誤射が全部有益だった証明ではない。
ただ、約0.28秒古い観測を操作する動的な戦場では、現在座標に対する`±80`のoracle自体が最適policyではない。
敵と照準が動くため、数字版の右側への早撃ちが、偶然の**雑な射撃lead**として働いた可能性がある。

もう一つの逆転はlane数である。V5は二台より一台のほうが`3.3 → 4.1`へ上がった。二laneは返答頻度を
増やすが、時間差の古い旋回が互いを上書きする。**脳を増やすと、古い命令の競合も増える。**

## TTLを振ると「鮮度の崖」が出た

V5のmodel、prompt、二T4、seedを固定し、観測から返答までのageがTTLを超えた命令だけ捨てた。
まず同じseed 7〜11の5本で比較した。

| TTL | kill平均 | 返答数 | accept | 期限切れ | 返答の静的正答率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 ms | **1.4** | 406 | 78 (19.2%) | 318 | **92.86%** |
| **300 ms** | **4.4** | 483 | 334 (69.2%) | 131 | 83.64% |
| 350 ms | 3.6 | 487 | 427 (87.7%) | 41 | 83.57% |
| 400 ms | 3.6 | 437 | 396 (90.6%) | 20 | 84.21% |

250 msでは正しい返答だけが残る方向へ寄ったのに、操作供給が足りず平均1.4まで崩れた。350〜400 msでは
供給量はあるが、古い命令も通る。今回の通信分布では、その間の300 msが最も良かった。

300 msだけ10 seedへ延長すると`[9,2,3,4,4,8,3,2,4,7]`、**46 kill・平均4.6**。
同じV5の二lane / 400 msは33、one lane / 400 msは41なので、それぞれ`+13`、`+5`だった。
数字版48には2 kill届かない。

これは300 msが普遍的な最適値という主張ではない。Quick Tunnelのjitter、15秒・10 seedという標本、
単一scenarioが残る。ただし少なくとも、TTLは単なる安全装置ではなくscoreを変える制御parameterであり、
**鮮度を厳しくしすぎても、緩くしすぎても弱い**ことが同じ身体で見えた。

paired差も過大評価しない。TTL 300−400 msは平均`+1.3 kill`、5勝3分2敗だが、paired bootstrap
95% percentile intervalは`[-0.4, 3.1]`、exact sign-flip testは`p=0.25`だった。one laneとの差
`+0.5`もinterval `[-1.2, 2.2]`。したがって「この10 seedでは300 msが最良」は言えるが、
**300 msの優越が統計的に確定した**とは言わない。

### TTL 300 msは一台でも効くのか

最後に、同じseed 7〜11でlane数とTTLを2×2にした。

| lane / TTL | kill平均 | 返答 | accept | 期限切れ |
| --- | ---: | ---: | ---: | ---: |
| 1 lane / 400 ms | **5.0** | 244 | 232 | 12 |
| 1 lane / 300 ms | 3.4 | 217 | 152 | 64 |
| 2 lane / 400 ms | 3.6 | 437 | 396 | 20 |
| **2 lane / 300 ms** | **4.4** | **483** | **334** | **131** |

一台のままTTLだけ短くすると、通る操作が232→152へ減って弱くなった。二台なら300 msでも334操作を
供給でき、二台400 msより古い尾を切りながら一台300 msの二倍以上を身体へ渡せた。したがって今回の
改善は「300 msという魔法の数字」より、**laneでaction supplyを稼ぎ、TTLでstale tailを削る組み合わせ**
として読むのが自然である。ただしこの2×2は5 seedなので、交互作用の確定には条件順をrandomizeした増量が要る。

## V6の手書き予測は失敗した

旧digit版の早撃ちを見て、右へ探索中なら右側のFIRE範囲を広げる非対称leadをpromptへ明記した。
簡潔版でも座標sweepは36 / 53、1 laneのpaired 3 seedはV5の`[8,2,8]`に対して
`[1,2,3]`。説明を足すと遠い左を右旋回へ反転する例まで出た。

よってV6は棄却した。遅延補償をするなら、自然言語で「未来を読め」と足すだけではなく、速度推定、
将来座標、action queueの状態を観測へ入れるか、明示的な予測器として別条件にする必要がある。

## 今回の結論

1. semantic actionを1 tokenへ符号化すれば、長いlabelの意味精度をほぼ保ったまま速度を回収できる。
2. 静的な意味正答率は、遅延した閉ループ制御のscoreを単調には予測しない。
3. lane追加はthroughputを増やすが、stale command競合まで増やし得る。
4. TTLには「古い命令を通す」と「操作を飢えさせる」の間に実測上のsweet spotがある。
5. 今回の最良semantic条件は2 T4 / TTL 300 msの4.6で、速いdigit版4.8とほぼ並んだが超えてはいない。

## 生ログ

SHA-256はWindows working treeのCRLFではなく、GitHubで配布される**Git LF blob**に対する値。

- [旧digit / 2 T4 / TTL 400 / 10本](results/colab-t4-digit-baseline-40ms-analyzed-10x-20260824.json) — SHA-256 `c9ab140ac0cc46ffd964cd482f9a7a7ce49f36e4cd7b031b4d414e7b4b58fd55`
- [action label / 2 T4 / TTL 400 / 10本](results/colab-t4-action-label-nf4-10x-20260824.json) — SHA-256 `8f1b796b97301e3ff455028d33e0484980c930b5b07e9e2770c08e83518537ef`
- [V5 / 2 T4 / TTL 400 / 10本](results/colab-t4-semantic-words-v5-nf4-10x-20260824.json) — SHA-256 `7bfa3d900ac7aa0c101c48cb07b4ebe5b3dfed23a72394b9f64072bc3a5d42ea`
- [V5 / 2 T4 / TTL 300 / 10本](results/colab-t4-semantic-words-v5-nf4-ttl300-10x-20260824.json) — SHA-256 `3a65e2f4f353dacaa742254fe8b87cc19c6d530b0fa332bc2d84681ea15f2767`
- [V5 / 2 T4 / TTL 250 / 5本](results/colab-t4-semantic-words-v5-nf4-ttl250-5x-20260824.json) — SHA-256 `0118a5c95f5f80d895e8c7fd8feef4e2c9b4f615497d30d25a5fc3a46485d381`
- [V5 / 2 T4 / TTL 350 / 5本](results/colab-t4-semantic-words-v5-nf4-ttl350-5x-20260824.json) — SHA-256 `3116989eaf0683432a52c473c0c13726197fa5da65da36e77cbd8864bf36595b`
- [V5 / 1 T4 / 10本](results/colab-t4-semantic-words-v5-nf4-one-lane-10x-20260824.json) — SHA-256 `aee7a7efa007deb1d311ddfb69cfbca5802831b4c70f216496e7ea3ab6c035d2`
- [V5 / 1 T4 / TTL 300 / 5本](results/colab-t4-semantic-words-v5-nf4-one-lane-ttl300-5x-20260824.json) — SHA-256 `611ce1d54fb1f0eb0ad31dc9423733f515028404328e9402d2847b50f2277b30`
- [V5 53座標sweep](results/remote-motor-boundary-sweep-semantic-words-v5-nf4-c-20260824.json) — SHA-256 `2070ea5b02bef90140c8f0a5d8b05e23b47d447e4021c8726ea059fbe6dffab0`
- [V5 境界近傍29問](results/remote-motor-probe-neighborhood-semantic-words-v5-nf4-c-20260824.json) — SHA-256 `6dbbf41e1395a4007c28b83d1600085ee31fbd5daad065ca0198e9fb2a88d9ef`
- [V6 lead 53座標sweep](results/remote-motor-boundary-sweep-semantic-words-v6-lead-simple-nf4-c-20260824.json) — SHA-256 `fbd9b892092a8929d55fbed42cac9b1eb7317d838889e1895eb003347b457299`
- [V6 lead / 1 T4 / paired 3本](results/colab-t4-semantic-words-v6-lead-one-lane-3x-20260824.json) — SHA-256 `cf51db9daa364b335e53e8d1fa0b9f65874e6ba5312a3f601f9c25f6989b2fe1`
- [action label NF4 53座標sweep](results/remote-motor-boundary-sweep-action-label-nf4-c-20260824.json) — SHA-256 `a612f768125fb7ab4a9e4c22fa1601d9afca47443071f28eef5212321415f127`
- [action label NF4 境界近傍29問](results/remote-motor-probe-neighborhood-action-label-nf4-c-20260824.json) — SHA-256 `56bb69a82e1f767dd045ed7a9b5195ba309d183f5f9120f76657c84f5468ba1d`
- [action label int8 53座標sweep](results/remote-motor-boundary-sweep-action-label-int8-a-20260824.json) — SHA-256 `fee572d36db0fe22511e3b25eb59acb1a991a1b145c07018ee30ebdeafa2595b`
- [action label int8 境界近傍29問](results/remote-motor-probe-neighborhood-action-label-int8-a-20260824.json) — SHA-256 `19888facaab12303902532add179e0c15729a403f6bb375ff4cb26d2dad4a6ef`
- [action label NF4 境界顕微鏡18問](results/remote-motor-boundary-microscope-action-label-nf4-c-20260824.json) — SHA-256 `758dd7716e3ee41dd4ea70aa4d84381bbf05d56cd978329bf554ea27b487d6e2`
- [探索中の小probe一覧](results/colab-t4-policy-tuning-probes-20260824.json)
- [10 seedのpaired bootstrap / exact sign-flip](results/colab-t4-semantic-words-v5-paired-analysis-20260824.json)

runtime endpoint、bearer token、Hugging Face tokenは保存していない。
