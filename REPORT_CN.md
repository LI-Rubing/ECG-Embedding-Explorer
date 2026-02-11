# ECG Embedding Explorer

## 1. 依赖与运行

推荐 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖版本约束：

- `numpy>=1.24`
- `pandas>=2.0`
- `plotly>=5.18`
- `pyarrow>=12.0`
- `scikit-learn>=1.2`
- `streamlit>=1.33`
- `umap-learn>=0.5.5`

运行应用：

```bash
streamlit run ui/app.py
```

强烈建议：

- ECG 数据优先转换为 `parquet` 后再导入应用。
- 使用 `parquet` 可启用元数据读取、按行惰性加载，以及在大数据量下更稳定/更快的交互体验。

可直接使用以下命令转换：

```bash
ECG_PKL_PATH=data/df_ecg.pkl ECG_PARQUET_PATH=data/df_ecg.parquet python scripts/convert_ecg_to_parquet.py
```

## 2. 数据格式要求

支持 `parquet / csv / pkl`。

一次真实运行的数据规模示例：

- `df_embeddings`: `rows=21799`, `cols=14`
- `df_ecg`: `rows=21799`, `cols=13`

### df_embeddings（必需）

- 必须包含：`ecg_id`
- 至少一个 embedding 列，例如：
  - `mean_embedding`
  - `mean_global_embedding`
- embedding 一般为 `np.ndarray` 或 `list/tuple`，且长度一致（如 1024）

可选元数据示例：
- `patient_id`、`device_group`、`sampling_rate`、`label`、`split` 等

本次数据中已观察到的列：
- `mean_embedding`、`mean_global_embedding`、`ecg_id`、`patient_id`、`tor`、`sampling_rate`
- `leads_partition_name`、`leads_array_index`
- `discovery_tsne_raw_embedding_x`、`discovery_tsne_raw_embedding_y`
- `device_name`、`device_model`、`device_name_present`、`device_group`

说明：
- 当前数据里 `discovery_tsne_raw_embedding_x/y` 为全缺失（`all NaN`），建议使用在线计算的 PCA / t-SNE / UMAP 结果。

### df_ecg（必需）

- 必须包含：`ecg_id`
- 除 `ecg_id` 外的列均视为导联信号列：
  - 每列为单导联信号（如长度 5000）
  - 多导联按列存储在同一行中

补充说明：
- 支持重复导联列名（例如多列都叫 `element`）。
- 应用会将所有非 `ecg_id` 列统一作为导联信号处理。

## 3. 主要功能

- 加载 embeddings 与 ECG 数据
- 数据校验（join key / 缺失 ID / embedding 长度一致性）
- PCA / t-SNE / UMAP 降维
- 默认显示 2D embedding；可切换到 3D（`Output dims = 3`）
- t-SNE 的 `metric` 可通过下拉选择（不局限于 Euclidean）
- UMAP 的 `metric` 可通过下拉选择（无需手动输入）
- Plotly 散点图（hover 元数据 + 颜色映射）
- 右侧 ECG 面板联动显示 ECG 波形与 metadata
- 提供 `ecg_id Copy Assistant`，用于检索并复制长 ID

## 4. 交互说明

- 首次进入应用时，右侧 ECG 面板保持等待状态，直到选择 `ecg_id`。
- 在 3D 模式下，Streamlit 原生单击回传可能不稳定（与环境相关）。
- 稳定方案：使用右侧 `ecg_id Copy Assistant` 或 `Load ECG by ecg_id`。
