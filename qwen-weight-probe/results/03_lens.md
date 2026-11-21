# Phase 3: reading u from the weights alone

Logit lens: `E @ ((1+g_final) * v)` with the tied embedding E. Top 25 tokens each way.

## u15 (+u: what the original edited neurons wrote)

**+**: `呗` · ` first` · ` series` · ` or` · ` thereby` · ` f` · `지요` · `即可` · ` easily` · `而不` · ` thus` · ` suit` · `,` · ` fx` · `owitz` · ` rather` · ` fn` · `.api` · ` however` · ` appropriately` · ` align` · `嘛` · `开口道` · ` order` · ` side`

**−**: `...*` · `...**` · `urada` · `essin` · `ologe` · `kede` · `tant` · `而是因为` · `gangen` · `iscos` · `zada` · `cante` · `skim` · `atina` · `同期的` · `ossid` · `-**` · `市委常委` · `echs` · `scher` · `yip` · `gando` · `党委副书记` · `zare` · `是一场`

## u19

**+**: `...*` · `她还是` · `而是因为` · `他还是` · ` Assinar` · `...**` · `emies` · `urada` · `ossid` · `istou` · `ург` · `威夷` · ` hanem` · `heden` · `ändern` · `-**` · `slaught` · `各种各样的` · `党组成员` · `无反顾` · `党委委员` · `liku` · `五颜六色的` · `市委常委` · `iste`

**−**: ` first` · `对` · ` count` · `即可` · ` yields` · ` basic` · ` appropriately` · ` suit` · ` result` · ` exact` · ` value` · `R` · ` accordingly` · ` align` · `呗` · ` using` · ` numerical` · `相关` · ` discret` · `/s` · ` safety` · `与` · ` or` · `,` · ` results`

## control: random unit vector

**+**: `zymy` · `�` · `lo` · ` distant` · `所` · `olding` · `ivol` · `满` · `ือก` · `го` · `aiss` · `光荣` · ` che` · ` Modelo` · `гра` · `upt` · `靡` · `的是一个` · ` recul` · `uca` · `我是个` · ` Unknown` · `滿` · ` Sole` · `mog`

**−**: `抗衡` · `inton` · `ولية` · `functional` · `زية` · `发展战略` · `tein` · `atria` · `反渗透` · `ollect` · `治疗后` · `功能性` · `theid` · `restore` · `żyw` · `terapi` · `東西` · `nerg` · `وى` · `-functional` · `ippe` · `合力` · `فع` · `arez` · ` pina`

## control: unedited L15 down_proj column 3058

**+**: `abe` · `非同` · ` nghi` · `[res` · `estet` · `ypes` · `rah` · `orej` · `ventus` · `Res` · `-profile` · `生活` · `汰` · ` bet` · `传出` · `.live` · ` CONCE` · `ächt` · `yday` · `賽` · `정` · `abri` · `ype` · `עי` · ` Res`

**−**: `illon` · `而复` · `illons` · `ít` · `�` · `ilu` · ` campion` · `能为` · `ickle` · ` swings` · `ités` · `直接向` · `�` · `illées` · `لف` · `ITTLE` · ` descansar` · `�` · `ún` · `ков` · `нику` · `ottes` · `им` · `inen` · `向阳`

## Strongest writers of u15 (|cos| of output column with u15)

| layer | kind | col | cos | edited by p50c |
|---|---|---|---|---|
| 11 | down | 346 | +0.325 |  |
| 18 | out | 1792 | +0.321 |  |
| 17 | down | 163 | -0.307 |  |
| 17 | out | 855 | -0.293 |  |
| 1 | down | 8 | -0.292 |  |
| 16 | down | 228 | +0.291 |  |
| 14 | out | 1485 | +0.291 |  |
| 14 | down | 221 | -0.290 |  |
| 1 | out | 896 | -0.289 |  |
| 9 | down | 42 | +0.288 |  |
| 0 | out | 864 | +0.288 |  |
| 19 | down | 11 | +0.287 | yes |
| 15 | down | 1155 | -0.282 | yes |
| 1 | out | 936 | -0.279 |  |
| 1 | down | 1 | -0.277 |  |
| 0 | down | 23 | +0.277 |  |
| 11 | down | 331 | +0.275 |  |
| 16 | down | 138 | -0.275 |  |
| 22 | out | 1270 | -0.274 |  |
| 17 | down | 20 | +0.271 |  |
| 1 | out | 966 | -0.270 |  |
| 0 | down | 204 | -0.270 |  |
| 15 | down | 322 | -0.269 |  |
| 1 | out | 977 | +0.268 |  |
| 3 | out | 1972 | +0.267 |  |

Layer counts among the top 200 writers: L1.out: 40, L0.out: 31, L15.down: 11, L4.out: 9, L3.down: 9, L2.out: 9, L14.out: 7, L1.down: 6, L2.down: 6, L0.down: 5, L17.out: 4, L22.out: 4, L23.down: 4, L19.out: 4, L10.out: 4, L11.down: 3, L16.down: 3, L9.down: 3, L13.down: 3, L12.out: 3, L9.out: 3, L13.out: 3, L17.down: 2, L14.down: 2, L3.out: 2, L16.out: 2, L5.out: 2, L7.down: 2, L18.out: 1, L19.down: 1, L18.down: 1, L20.down: 1, L22.down: 1, L10.down: 1, L6.out: 1, L20.out: 1, L23.out: 1, L5.down: 1, L4.down: 1, L11.out: 1, L8.down: 1, L7.out: 1

## Strongest readers of u15 in layers 16-23 (MLP gate/up rows through the pre-MLP norm)

Rough max |cos| expected for random directions: 0.146

| layer | kind | neuron | cos |
|---|---|---|---|
| 17 | gate | 20 | -0.322 |
| 22 | gate | 89 | -0.309 |
| 22 | up | 103 | +0.304 |
| 22 | up | 1100 | -0.289 |
| 23 | gate | 78 | -0.254 |
| 16 | up | 438 | +0.240 |
| 17 | gate | 83 | -0.238 |
| 19 | up | 11 | +0.213 |
| 19 | gate | 11 | -0.211 |
| 22 | gate | 1217 | -0.209 |
| 16 | gate | 265 | -0.208 |
| 19 | up | 343 | +0.204 |
| 23 | up | 78 | +0.202 |
| 16 | up | 228 | -0.202 |
| 18 | up | 5 | +0.201 |
| 23 | gate | 284 | -0.196 |
| 23 | up | 161 | -0.195 |
| 22 | gate | 1 | -0.193 |
| 23 | gate | 218 | -0.190 |
| 17 | gate | 39 | -0.190 |
