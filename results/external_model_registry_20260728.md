# External time-series model registry and provenance audit

日期：2026-07-28  
用途：ROGII Heel-Registered Sparse-Anchor Student residual portfolio  
狀態：source/provenance audit 完成；architecture-only CPU implementation 已執行；**沒有下載任何外部 weights 或外部訓練資料**

## 1. Executive decision

目前乾淨、互補且可在本機重現的主 portfolio 是：

1. **Paired PatchTST-derived cross-encoder**：多解析度局部 pattern 與顯式
   horizontal/typewell interaction。
2. **S4D-inspired diagonal long-convolution encoder**：長距離 slope、whole-well
   state 與 repeated-bed context。
3. **1D FNO**：低頻 smooth residual field。
4. **TCN**：便宜的 dilated-convolution architecture control。

以上四個 family 均以 random initialization、competition fold-local data、純 PyTorch
實作；沒有外部資料或 pretrained weights。2026-07-28 的執行環境是 Windows、
Python 3.12.10、**PyTorch 2.12.1+cpu**。實作不使用 custom CUDA kernels，CPU
serialization/reload 與 finite-output checks 已通過。

預訓練模型中：

- **MOMENT-small/base 不可描述為沒有 geology exposure**。其 Time-series Pile
  明確包含 UCR `Rock` 岩石光譜資料。這不是 well log、GR 或競賽資料的已知直接
  overlap，但屬 geology-adjacent pretraining exposure。
- **TTM R2** 是最便宜且來源相對透明的 pretrained scouting candidate。
- **Chronos-Bolt-tiny** 僅適合低成本 frozen probe；Bolt 的精確 training corpus
  未完整披露。
- **TimesFM 2.5** 的 weights 小於 1 GB，但其 GiftEvalPretrain corpus 約
  975 GB，另有未完整列舉的 synthetic/augmented data；不先做大型 adapter。
- **Mamba text weights 不採用**。它們是 text-Pile weights，input stem、task 與
  provenance 都不適合本問題。S4D 與 Mamba 只算一個長序列 family。

ChronoLog GR 因本專案已重現高度 correlation／可能 overlap 風險，**禁止進入
training、pretraining、retrieval、feature construction 或 model selection**。

## 2. Audit method and evidence boundary

- 只使用官方論文、作者／官方 repository、PyPI metadata 與 Hugging Face
  model/dataset cards。
- Repository SHA 與 Hub revision 是 2026-07-28 查詢時的 snapshot；實際執行必須
  pin SHA/revision，不依賴浮動的 `main`。
- 下列 weight SHA256 與 byte size 來自官方 Hub file metadata。**本次沒有把 weight
  files 下載到工作區。**
- Architecture-only 模型沒有 canonical parameter count；其實際 parameter count
  必須由每個 run manifest 的 instantiated model 計算，不可引用不相同的論文配置。
- Model/repository license 不自動解決所有上游 pretraining dataset 的授權；資料來源
  不完整時標記為 unresolved。
- 「沒有找到 well/geology 名稱」不是 absence proof。只有明確列舉的 corpus 才能作
  有限度的 negative evidence。
- External source audit 不是模型品質證據；是否進入 full OOF 仍以 matched quick
  experiment、negative controls 與 strict fold-local validation 決定。

## 3. Reproducible source snapshots

| Family / package | Official source snapshot | Release | License |
|---|---|---|---|
| PatchTST | [`yuqinie98/PatchTST@204c21e`](https://github.com/yuqinie98/PatchTST/commit/204c21efe0b39603ad6e2ca640ef5896646ab1a9) | 無 official tags/releases | Apache-2.0 |
| S4 / S4D | [`state-spaces/s4@e757cef`](https://github.com/state-spaces/s4/commit/e757cef57d89e448c413de7325ed5601aceaac13) | [`v4.1.0@976c9e9`](https://github.com/state-spaces/s4/commit/976c9e9d07b64ae1cdd4219f5b660eefa6a380ff) | Apache-2.0 |
| Mamba | [`state-spaces/mamba@e9594ce`](https://github.com/state-spaces/mamba/commit/e9594ce1c732d97440f0332fdc43170a2294dbfa) | [`mamba-ssm==2.3.2.post1`](https://pypi.org/project/mamba-ssm/2.3.2.post1/), tag SHA `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` | Apache-2.0 |
| NeuralOperator | [`neuraloperator/neuraloperator@86a8bc7`](https://github.com/neuraloperator/neuraloperator/commit/86a8bc7812a31b42c4f7895693cf4ac11521c066) | [`neuraloperator==2.0.0`](https://pypi.org/project/neuraloperator/2.0.0/), tag SHA `b307030d9b7ee530823d6a05bddb8238ba22a743` | MIT |
| MOMENT | [`moment@38f7310`](https://github.com/moment-timeseries-foundation-model/moment/commit/38f7310ad594100747ca2a8357e9c7ca7d323e0e) | [`momentfm==0.1.4`](https://pypi.org/project/momentfm/0.1.4/) | MIT |
| TS2Vec | [`zhihanyue/ts2vec@b0088e1`](https://github.com/zhihanyue/ts2vec/commit/b0088e14a99706c05451316dc6db8d3da9351163) | 無 official release | MIT |
| TCN | [`locuslab/TCN@2f8c2b8`](https://github.com/locuslab/TCN/commit/2f8c2b817050206397458dfd1f5a25ce8a32fe65) | 無 official release | MIT |
| Granite TSFM / TTM | [`ibm-granite/granite-tsfm@31c7a6b`](https://github.com/ibm-granite/granite-tsfm/commit/31c7a6bd640c41e53777c0afe3a13d24e00f7fe7) | [`granite-tsfm==0.3.6`](https://pypi.org/project/granite-tsfm/0.3.6/), tag SHA `c2c7b3771bffd9b7e101891607225368f01e5e93` | Apache-2.0 |
| TimesFM | [`google-research/timesfm@3dae50b`](https://github.com/google-research/timesfm/commit/3dae50b20d7a724981e8ea36cda75578f80dd2dc) | [`timesfm==2.0.2`](https://pypi.org/project/timesfm/2.0.2/), tag SHA `4a6c5cdae7ac73a450843e035fbfc3ffa08e9caf` | Apache-2.0 |
| Chronos | [`amazon-science/chronos-forecasting@7dc4435`](https://github.com/amazon-science/chronos-forecasting/commit/7dc4435706a4454feb79df44ca9f33631f3027bf) | [`v2.3.1`](https://github.com/amazon-science/chronos-forecasting/releases/tag/v2.3.1) | Apache-2.0 |

PyPI archive hashes沒有保存於此 audit；若未來安裝這些 packages，應由 lockfile 另外
固定 wheel/sdist SHA256。上表的 code SHA 與下表的 weight SHA256 是本 registry 的
可重現 identity。

## 4. Exact pretrained weight identities

| Model | Hub revision | Parameters | Weight file | SHA256 | License / local status |
|---|---|---:|---:|---|---|
| MOMENT-1-small | [`411e288267f82cce86296dbe4d6c8bc533cc162f`](https://huggingface.co/AutonLab/MOMENT-1-small/tree/411e288267f82cce86296dbe4d6c8bc533cc162f) | 37.9M | 151,615,328 B safetensors | `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877` | MIT; not downloaded |
| MOMENT-1-base | [`5e44b0ea26376a176360f87831124e018f876d96`](https://huggingface.co/AutonLab/MOMENT-1-base/tree/5e44b0ea26376a176360f87831124e018f876d96) | 113.5M | 453,940,120 B safetensors | `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825` | MIT; not downloaded |
| Granite TTM R2 main | [`d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/tree/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f) | 805,280 | 3,240,592 B safetensors | `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710` | Apache-2.0; not downloaded |
| TimesFM 2.5 200M PyTorch | [`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3) | 231.3M HF metadata | 925,181,104 B safetensors | `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8` | Apache-2.0; not downloaded |
| Chronos-Bolt-tiny | [`a0e552de83495b5c28c14c71c374f3e33280b340`](https://huggingface.co/amazon/chronos-bolt-tiny/tree/a0e552de83495b5c28c14c71c374f3e33280b340) | 8.7M | 34,622,352 B safetensors | `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32` | Apache-2.0; not downloaded |
| Mamba-130M text reference | [`c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0`](https://huggingface.co/state-spaces/mamba-130m/tree/c4dcd5d65dff7d0bff5a4c97e30d82e57190f1b0) | nominal 130M | 516,618,642 B PyTorch bin | `29a023467229397eb03fd31e088c08c9193b734249f0c07d51af03873162e723` | Apache-2.0; rejected, not downloaded |

MOMENT 論文的 approximate counts 是 small/base 約 40M/125M；本 registry 使用實際
Hub artifact metadata 的 37.9M/113.5M。TimesFM 的名稱為 200M，但官方說明另有
約 30M quantile head；HF artifact metadata 是 231.3M。

上述每個單一 weight artifact 都小於 5 GB，可以在使用者授權範圍內個別下載；本輪
仍選擇不下載，因為尚未有 matched quick signal。大型 training datasets 不在下載
範圍內。

## 5. High-priority architecture-only families

### 5.1 Paired PatchTST-derived multi-resolution cross-encoder

- **Primary sources**：[PatchTST paper](https://arxiv.org/abs/2211.14730)；
  [official code](https://github.com/yuqinie98/PatchTST)。
- **Canonical contribution**：PatchTST 提供 patching 與 channel-independent
  processing。顯式 horizontal/typewell cross-attention 是本專案新增，因此正式名稱
  必須使用 **PatchTST-derived paired cross-encoder**，不可說是原版 PatchTST。
- **Parameters/download**：依 patch sizes、hidden width、layers 而定；沒有通用
  pretrained checkpoint，external download 為 0。
- **Runtime**：純 PyTorch Conv1d/Transformer/MultiheadAttention；目前
  `torch==2.12.1+cpu` 可離線執行，無 custom kernels。
- **Pretraining/provenance**：官方 repository 只提供在指定資料集做 self-supervised
  pretraining 的流程，沒有可泛化到本問題的官方 generic weights。若做 masked
  pretraining，只能使用 outer-training wells。
- **Architecture/pretrained separation**：
  - A：supervised random-init architecture-only。
  - B：同 architecture 的 fold-local masked pretraining。
  - 無外部 pretrained C。
- **Representation/diversity**：horizontal GR 與 typewell GR 的多解析度 local
  motifs、lag-tolerant correspondence 與 cross-series interaction；預期直接改善
  centered-shape，和 incumbent/common-mode registration 的 residual diversity 高。
- **Required controls**：no-cross、wrong-typewell、pair shuffle、GR-order/phase
  destroy。No-cross 必須保留 matched head shape/capacity。
- **Status**：**implemented and CPU-tested, high-priority quick screen**。

### 5.2 S4D-inspired diagonal long convolution

- **Primary sources**：[S4D paper](https://arxiv.org/abs/2206.11893)；
  [Annotated S4D by the authors](https://srush.github.io/annotated-s4/s4d.html)；
  [S4 repository](https://github.com/state-spaces/s4)。
- **Parameters/download**：config-dependent；作者教學中的小例子約 200K，但不是
  本專案 canonical count。無 time-series pretrained weights，external download 為 0。
- **Runtime**：本專案只採 diagonal kernel + `torch.fft` 的純 PyTorch版本；CPU
  offline 可重建，不載入 S4 optional CUDA kernels。
- **Pretraining/provenance**：無外部資料；A 為 random-init architecture-only。
  Fold-local self-supervision只有在 quick signal 後才值得測。
- **Representation/diversity**：全井長距 state、slope、slow drift 與 repeated-bed
  context；與 patch attention、FNO 和 TCN 有清楚不同的 inductive bias。
- **Status**：**implemented and CPU-tested, high-priority quick screen**。

### 5.3 Small Mamba

- **Primary source**：[official repository](https://github.com/state-spaces/mamba)。
- **Runtime**：官方 optimized path 假設 Linux、NVIDIA GPU、CUDA 11.6+ 與
  selective-scan/custom kernels；這不是目前 Windows CPU 的乾淨 dependency。
- **Pretrained weights**：官方 `mamba-130m` 是約 300B text-Pile tokens 的語言模型
  weight；序列 input stem 與任務不相容，web/code corpus 也無法提供本問題需要的
  clean provenance。
- **Architecture/pretrained separation**：只允許之後建立 small pure-PyTorch
  random-init control；不使用 text weights。
- **Representation/diversity**：與 S4D 都是長序列 state-space family。即使日後有
  signal，也不可把 S4D 和 Mamba 當成兩個獨立 ensemble votes。
- **Status**：**deferred/rejected for current CPU portfolio**；S4D 是乾淨 fallback。

### 5.4 1D FNO / NeuralOperator

- **Primary sources**：[NeuralOperator repository](https://github.com/neuraloperator/neuraloperator)；
  [official FNO theory guide](https://neuraloperator.github.io/dev/theory_guide/fno.html)。
- **Parameters/download**：width、layers、Fourier modes 決定；無通用 pretrained
  weights，external download 為 0。
- **Runtime**：純 PyTorch FFT；CPU offline 可執行。
- **Pretraining/provenance**：只引用 architecture，不使用 `neuralop.data` 或 bundled
  physics datasets。官方 library 中名為 “The Well” 的 physics dataset 也不進入本
  專案；名稱不能被誤解成已使用 well-log data。
- **Unequal-length handling**：每井 deterministic resampling 到 normalized-MD
  uniform grid，提供 coordinate/spacing/mask；輸出再插值回原 prediction rows。
  Resampling 不得學習 validation statistics。
- **Representation/diversity**：低 Fourier modes 直接偏向 smooth residual field，
  特別適合 zero-mean smooth-shape component；與 PatchTST/S4D diversity 高。
- **Known risk**：periodic wrap/boundary artifact；需要 right padding、coordinate
  channel、terminal metric 與 shape mean constraint。
- **Status**：**implemented and CPU-tested, high-priority quick screen**。

## 6. Lightweight representation controls

### 6.1 TS2Vec

- **Primary sources**：[paper](https://arxiv.org/abs/2106.10466)；
  [official repository](https://github.com/zhihanyue/ts2vec)。
- **Parameters/runtime**：官方 default hidden 64、depth 10、representation 320；
  依官方 source 對 1–2 input channels 計算約 0.64M trainable parameters。這是計算值，
  不是官方 model-card count。Repository 的舊 environment 使用 PyTorch 1.8.1；
  應在 current PyTorch 重新實作必要 encoder。
- **Weights/provenance**：無 generic pretrained weights；所有 contrastive pretraining
  必須 fold-local。
- **Matched comparison**：supervised random-init、random-frozen、fold-local TS2Vec
  SSL 使用同 stem/head/capacity。
- **Representation/diversity**：hierarchical local/global convolutional contrastive
  features；paired head 可使用 difference/product/cross-correlation。
- **Status**：**control queued**；先看更便宜 TCN 是否有 shape signal。

### 6.2 TCN

- **Primary sources**：[paper](https://arxiv.org/abs/1803.01271)；
  [official repository](https://github.com/locuslab/TCN)。
- **Parameters/download/runtime**：config-dependent、無 weights；residual dilated
  Conv1d，純 PyTorch CPU offline。
- **Representation/diversity**：便宜的 local motif/slope control，預期 diversity
  中等。若採用非 causal full-sequence convolution，正式名稱需標示
  **TCN-derived**。
- **Status**：**implemented and CPU-tested, cheap control**。

## 7. Pretrained foundation candidates

### 7.1 MOMENT-small/base

- **Primary sources**：[MOMENT paper](https://arxiv.org/html/2402.03885)；
  [official repository](https://github.com/moment-timeseries-foundation-model/moment)；
  [Time-series Pile](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)。
- **Model architecture**：fixed input length 512、patch length 8、64 patches、30%
  masked modeling。Paper 的 small/base 分別為 6/12 layers、hidden 512/768、
  8/12 heads。
- **Runtime**：`momentfm==0.1.4`、Python >=3.10，standard PyTorch/Transformers；
  weights cached 後可 CPU offline，但 base latency/memory 高於本地 tiny models。
- **Pretraining corpus**：Time-series Pile current Hub revision
  [`ea89753da2b451928436adb333c7a2e892461c7d`](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/tree/ea89753da2b451928436adb333c7a2e892461c7d)，
  dataset card 報告 13M unique series、1.23B timestamps、13 domains；來源包括
  Informer、Monash、UCR/UEA、TSB-UAD。HF stored size 為 8.71 GB，card 另描述
  約 20.085 GB 原始資料。
- **Revision limitation**：MOMENT checkpoint/model card 沒有 pin 實際 pretraining
  時所用的 Time-series Pile revision，故 weight-to-corpus exact mapping unresolved。
- **License limitation**：Pile top-level 標記 MIT；它聚合多個上游 archives，不能由
  top-level tag 推定每個 constituent dataset 的授權已完整清理。

#### MOMENT `Rock` geology-adjacent caveat

MOMENT 論文的完整 Pile metadata 明確列出 `Rock`；[UCR/UEA `Rock` 官方資料頁](https://www.timeseriesclassification.com/description.php?Dataset=Rock)
說明：

- 70 條長度 2,844 的 ASTER spectral-library curves；
- 每條是 rock spectral reflectance 對 wavelength，**不是時間序列**；
- 四類為 mafic、quartzite、marble、schist；
- official split 為 train 20 / test 50。

MOMENT 論文說 pretraining 只使用各資料集 training splits，因此合理推論 checkpoint
可能看過 `Rock` 的 20 條 official-training curves，而不是 70 條全部。這仍是明確的
geology-adjacent spectral exposure。

稽核結論：

- 尚無證據顯示 `Rock` 是 gamma ray、well log、ROGII competition data 或同源 wells。
- 也不能聲稱 pretrained MOMENT「完全沒有 geology exposure」。
- MOMENT architecture-only 是乾淨 control。
- Pretrained frozen/adapter 結果必須標示 **provenance-exposed**，並先確認競賽規則
  是否接受此類外部 pretraining。

- **Matched comparison**：
  1. same-config supervised random-init；
  2. random-frozen encoder + learned fold-local head；
  3. pretrained-frozen encoder + identical head；
  4. 只有第 3 項在 pairing-sensitive controls 下明顯勝過前兩項，才做 adapter。
- **Representation/diversity**：masked-patch general time-series embedding，但和
  PatchTST 都屬 patch transformer，cross-family diversity 只屬中等。
- **Status**：**pretrained route held for provenance decision**。若獲准，只先測
  small；small 無 transfer signal 則停止，不下載 base。Pile 超過 5 GB，沒有理由
  下載，且下載前需要另行詢問。

### 7.2 Granite Tiny Time Mixer R2

- **Primary sources**：[TTM paper](https://arxiv.org/abs/2401.03955)；
  [R2 model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)；
  [official repository](https://github.com/ibm-granite/granite-tsfm)。
- **Artifact**：main config context 512、prediction 96、patch 64、`d_model=192`、
  2 layers、3 adaptive patch levels；805,280 parameters，約 3.24 MB。
- **Runtime**：`granite-tsfm==0.3.6` 要求 Python >=3.11；IBM 明確支援 CPU/laptop，
  weight cached 後可完全離線。
- **Disclosed training data**：Australian electricity/weather、Bitcoin、KDD 2018、
  London smart meters、Saugeen River Flow、solar/sunspots、births、wind、PEMS
  traffic、LOS_LOOP。官方清單未列 well log、GR 或 geology dataset。
- **Overlap limitation**：未列出直接 geology source 是有利 evidence，但不是沒有
  overlap 的證明。
- **Architecture/pretrained separation**：官方 `get_model()` 支援
  `random_init_small/medium/large`，可做乾淨 same-architecture comparison。
- **Representation/diversity**：adaptive patch/mixer 與 optional cross-channel
  decoder；比 transformer attention 便宜，但仍是 forecasting-first，sampling
  frequency semantics 與 depth-domain inverse alignment 不一致。
- **Status**：**highest-priority pretrained quick probe**；frozen backbone →
  matched random-init，若 wrong pairing 不降低收益即停止。

### 7.3 TimesFM 2.5

- **Primary sources**：[official repository](https://github.com/google-research/timesfm)；
  [2.5 model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)；
  [GiftEvalPretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)；
  [GIFT-Eval paper](https://arxiv.org/abs/2410.10393)。
- **Artifact/runtime**：HF metadata 231.3M parameters、925,181,104-byte weights；
  model card 支援 context up to 16K。Python >=3.10；PyTorch/Flax，cached 後可
  CPU/GPU offline，但 CPU probe 明顯重於其他 candidates。
- **Pretraining provenance**：TimesFM 2.5 model card 明列
  `Salesforce/GiftEvalPretrain` 加「Synthetic and augmented data」。
  GiftEvalPretrain audited revision 是
  [`6830b624de7ed2b3d3e5b85bb6959d81dcc5d874`](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain/tree/6830b624de7ed2b3d3e5b85bb6959d81dcc5d874)，
  card 報告 71 univariate + 17 multivariate datasets、七 domains、13 frequencies、
  4.5M series、230B points、約 975 GB。
- **Overlap audit**：official Hub filenames 未發現 `well`、`geology`、`seismic`、
  `rock`、`oil`、`drill` 或 `gamma`；但巨量 corpus 的 upstream contents/license
  與 synthetic/augmentation composition 沒有完整列舉，因此 provenance 為
  **incomplete, not cleared**。
- **Architecture/pretrained separation**：231M random-init full training 不符合
  low-cost matched control。最多只能做 pretrained-frozen 與 same-config
  random-frozen embedding probe；這不能單獨證明 architecture gain。
- **Representation/diversity**：large decoder-style forecasting representation；
  任務不匹配，且與其他 patch/forecasting models 的 residual diversity 預期低。
- **Status**：**low-priority scouting only**；TTM/Chronos 無 signal 時不啟動。
  GiftEvalPretrain 不下載。

### 7.4 Chronos-Bolt-tiny

- **Primary sources**：[Chronos paper](https://arxiv.org/html/2403.07815)；
  [official repository](https://github.com/amazon-science/chronos-forecasting)；
  [Bolt-tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny)。
- **Artifact/runtime**：8.7M parameters、34,622,352-byte weights；standard
  PyTorch/Transformers，Python >=3.10，CPU/GPU offline。
- **Provenance**：Bolt card 只說使用「nearly 100 billion time series
  observations」，未完整枚舉 corpus。原始 Chronos 論文列 28 個 datasets
  （約 890K series、84B observations）、10M TSMixup 與 1M KernelSynth，領域包括
  energy、weather、transport、retail、web；沒有明列 geology/well logs。
- **Evidence boundary**：不能假定 Bolt 的 exact corpus 等同原始 Chronos corpus；
  因此 direct geology overlap 是 unknown，不是 cleared。
- **Architecture/pretrained separation**：可用同 config random initialization 做
  frozen/random-init control；只有 pairing-sensitive quick signal 才考慮 adapter。
- **Representation/diversity**：patch-based T5/direct quantile forecasting；
  和 paired PatchTST 有部分重疊，預期 diversity 低至中。
- **Status**：**second low-cost pretrained probe after TTM**。

## 8. ChronoLog external-GR prohibition

檢查來源：

- [ChronoLog data repository](https://github.com/zsylvester/chronolog_data)
- [Sylvester 2023 project/paper page](https://zsylvester.github.io/papers/chronolog/)

本專案先前對 681-well Permian Basin GR 重做 overlap audit；raw、linear-detrended、
first-difference 最大 Pearson 分別為約 `0.973114 / 0.959059 / 0.857007`。這無法
排除來源 overlap 或高度相似低頻 shape。舊的較低 correlation screen 已撤回。

因此 ChronoLog：

- 不可作 supervised/unsupervised pretraining data；
- 不可作 nearest-neighbor index、retrieval library 或 feature bank；
- 不可用來調 representation、normalization 或 ensemble weights；
- 不因 architecture-only 模型需要「更多資料」而解除禁令。

這項禁令不影響使用公開 architecture/code；它只禁止外部 ChronoLog curves/data。

## 9. Architecture-only versus pretrained comparison matrix

| Family | A: random-init architecture | B: random/fold-local control | C: external pretrained | Decision rule |
|---|---|---|---|---|
| Paired PatchTST-derived | Yes | fold-local masked pretraining | 無 generic official weight | B 必須勝過 A 且 pairing controls 有效 |
| S4D | Yes | optional fold-local SSL | None | 先以 A 判斷 long-state signal |
| Mamba | Deferred | pure-PyTorch small control only | text weights rejected | 不與 S4D 重複計權 |
| FNO1D | Yes | random-frozen optional | None | 看 zero-mean shape 與 terminal tail |
| TCN | Yes | random-frozen | None | cheap capacity/control baseline |
| TS2Vec | Yes | fold-local contrastive SSL | None | SSL 必須勝過 same-capacity A |
| MOMENT | Yes | random-frozen | frozen pretrained, provenance-exposed | C 勝過 A/B 且 shuffle gain 消失才算 transfer |
| TTM R2 | Yes, official helper | random-frozen | frozen pretrained | 低成本第一個 TSFM probe |
| TimesFM 2.5 | full A 成本過高 | random-frozen only | frozen pretrained | 只能稱 embedding signal，不能稱完整 architecture gain |
| Chronos-Bolt-tiny | Yes | random-frozen | frozen pretrained | TTM 後低成本 probe |

所有比較必須使用相同 outer folds、input preprocessing、target decomposition、
head capacity、training budget 與 row-count weighting；input scaler、target scaler、
head、adapter、learned representation 均只能 fit outer-training wells。

## 10. Portfolio diversity and status decision

| Family | Expected new residual representation | Expected diversity vs incumbent | Current status |
|---|---|---|---|
| Paired PatchTST-derived | multi-resolution cross-series local shape | High | Implemented; quick screen |
| S4D | long state, slope, repeated context | High | Implemented; quick screen |
| FNO1D | low-frequency zero-mean smooth field | High | Implemented; quick screen |
| TCN | cheap dilated local/slope control | Medium | Implemented; control |
| TS2Vec | hierarchical contrastive local/global shape | Medium–high | Queued after TCN |
| MOMENT-small | general masked-patch embedding | Medium; overlaps PatchTST | Held for provenance decision |
| TTM R2 | adaptive patch/mixer | Medium–low | First pretrained probe |
| Chronos-Bolt-tiny | patch/T5 forecast representation | Low–medium | Second pretrained probe |
| TimesFM 2.5 | large forecasting representation | Low | Last-resort scouting |
| Mamba | selective long state | High but duplicates S4D family | Deferred |

停止規則：

- wrong-typewell、pair shuffle 或 phase-destroying control 保留同等收益：停止；
- pretrained 不勝過 same-capacity random-init/random-frozen：不得宣稱 transfer；
- quick signal 只來自 common-mode、沒有 slope/shape 改善：不進 full OOF；
- catastrophic tail、terminal RMSE 或 fold stability 明顯惡化：不進 ensemble；
- 同一 long-sequence 或 patch-transformer family 不因 seeds/variants 較多而重複加權。

## 11. Offline, download and execution record

- Architecture-only execution：paired PatchTST-derived、S4D-inspired、FNO1D、TCN。
- Runtime：Python 3.12.10，PyTorch `2.12.1+cpu`，Windows，float32，無 CUDA/custom
  kernels。
- Focused implementation tests：8 passed；包含 deterministic same-seed output、
  no-cross matched capacity、fold-local row-weighted scaling、random-frozen encoder、
  finite fail-closed checks、CPU serialization/reload 與 state digest。
- External model weights downloaded：**none**。
- External training datasets downloaded：**none**。
- Paid cloud/GPU used：**none**。
- Kaggle submission created/uploaded：**none**。
- Offline inference：所有入選 architecture-only family 已可完全離線。TTM、MOMENT、
  TimesFM、Chronos 只有在未來先把 pin 的 weight artifact cache 到本機後才能離線；
  inference 本身不需要網路。
- >5 GB artifacts：MOMENT Time-series Pile、GiftEvalPretrain 都超過門檻，未下載；
  也沒有目前實驗上的必要性。

## 12. Unresolved risks

1. MOMENT weight 沒有 pin 它實際使用的 Time-series Pile revision，且有明確
   `Rock` geology-adjacent exposure。
2. TimesFM 2.5 的 synthetic/augmented composition 與 GiftEvalPretrain upstream
   licenses 沒有完整 weight-level traceability。
3. Chronos-Bolt 的 exact nearly-100B corpus 沒有完整列舉。
4. TTM 的 corpus 較透明，但「未列 geology」仍不是 absence proof。
5. Foundation forecasting objectives與 inverse cross-instrument alignment 不同；
   即使 frozen embeddings 有 signal，也必須由 pairing-sensitive controls 證明。
6. External model license 不等於所有 pretraining source licenses；任何最終使用仍需
   在 competition rule、fair availability 與 offline requirements 下再確認。

目前最安全的執行結論仍是：先用已完成的四個 architecture-only family 做 matched
quick experiments；在它們證明 slope/shape signal 前，不下載 foundation weights。
