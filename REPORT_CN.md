# ECG Embedding Explorer

## 1. 项目概述

该项目是一个基于 Streamlit 的 ECG embedding 交互式分析工具，核心目标是把「embedding 可视化」和「单条 ECG 波形回看」串成一个联动工作流。

当前代码实现已覆盖：
- 多格式数据加载（本地路径 + 文件上传）
- embedding 降维与 2D/3D 可视化
- 点选/检索 `ecg_id` 后按需加载 ECG
- 面向大样本的性能保护（抽样、缓存、降级策略、预取上限）

## 2. 运行方式

推荐 Python 3.10+。

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

ECG 数据建议优先转为 parquet（更适合按行读取）：

```bash
ECG_PKL_PATH=data/df_ecg.pkl ECG_PARQUET_PATH=data/df_ecg.parquet python scripts/convert_ecg_to_parquet.py
```

转换脚本支持环境变量：
- `ECG_PKL_PATH`
- `ECG_PARQUET_PATH`
- `ECG_PARQUET_COMPRESSION`（默认 `snappy`）
- `ECG_PARQUET_CHUNK_SIZE`（默认 `2000`）

## 3. 数据与校验规则

支持 `parquet / csv / pkl`。

`df_embeddings` 要求：
- 必含 `ecg_id`
- 至少有一列 array-like embedding（如 `mean_embedding` / `mean_global_embedding`）
- 同一 embedding 列内长度应一致

`df_ecg` 要求：
- 必含 `ecg_id`
- 其余列视为导联信号列（每列通常是 array-like waveform）

应用内置校验包含：
- join key 是否存在
- embedding 列是否存在、是否为 array-like、长度是否一致
- 可选 Full Validation：缺失 ID 检查、各列缺失统计、数据摘要

## 4. 当前实现的关键功能

- **加载层**
  - 支持 sidebar 输入路径与上传文件两种入口
  - `df_ecg` 可选择不整表入内存，仅读取 parquet 元数据
  - 需要时可切换为整表加载（高内存模式）

- **降维与可视化**
  - 支持 3 种来源：
    - 预计算 2D 列（`tsne_mean_embedding_x/y`）
    - 手动选择任意数值列作为 X/Y
    - 高维 embedding 在线降维（PCA / t-SNE / UMAP）
  - 输出维度可选 2D 或 3D
  - PCA 支持自定义 PC 轴映射（X/Y/Z 对应第几个主成分）
  - Plotly 使用 `webgl` 渲染，支持 hover 元数据与颜色映射

- **性能与稳定性策略**
  - `st.cache_data` 缓存：
    - 数据加载
    - 降维结果
    - PCA 组件计算
    - ECG 行级读取与批量预取
  - 大样本保护：
    - `LARGE_EMBEDDING_THRESHOLD=10000`
    - 样本较大且方法为 t-SNE/UMAP 时，默认自动回退到 PCA（可手动允许慢速）
  - ECG 预取上限：
    - `MAX_ECG_PREFETCH_POINTS=1200`
    - 超限时跳过预取以保证交互流畅

- **ECG 联动面板**
  - 左侧散点图选择后，右侧加载对应 ECG 与 metadata
  - 未整表加载时，按 `ecg_id` 从 parquet 按行读取
  - 包含 `ecg_id Copy Assistant`（子串搜索 + 复制）
  - 提供 `Load ECG by ecg_id`（手动输入 / 下拉选择）

## 5. 已知限制与现状判断

- Streamlit 在 3D 单点点击回传上仍可能不稳定，代码已提供文本检索与手动加载作为稳定 fallback。
- 点击模式在低内存机器、较大点数下稳定性较差；UI 中已对较大 sample 给出提醒与限制建议。
- Full Validation 的缺失统计在超大数据集上仍可能耗时。
- 目前未包含系统化单元测试与基准测试，质量保障主要依赖交互验证。

## 6. 下一步建议

- 增加 `data/loader.py` 与 `embeddings/reducer.py` 的单元测试（边界输入、异常路径、参数兼容）。
- 为大样本场景补充性能基准（PCA/t-SNE/UMAP 在不同 N 下耗时与内存曲线）。
- 在 UI 中增加更明确的计算状态提示（例如分步骤进度与预计耗时）。
