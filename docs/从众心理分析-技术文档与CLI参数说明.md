### 从众心理分析 - 技术文档与 CLI 参数说明

本文档基于优化后的四个核心模块：`data_cleaning_optimized.py`、`like_analysis_optimized.py`、`sentiment_analyzer_optimized.py`、`similarity_analysis_optimized.py`，系统解读当前从众心理分析算法的实现思路、输入输出、可视化与降级策略，并补充统一入口 `text_analysis_unified.py` 的完整参数说明与推荐使用流程。

---

### 一、总体设计与数据流
- **统一入口**：`text_analysis/text_analysis_unified.py`
  - 子命令模块：`cleaning`（数据清洗）、`time`（时间分析）、`like`（点赞分析）、`sentiment`（情感分析）、`similarity`（文本相似度）。
  - 通用参数：`--video-id`、`--limit`、`--use-cleaned-data`、`--cleaned-data-path`、`--test`、`--no-save`、`--no-report`、`--no-viz`。
- **数据源**：
  - MySQL 表 `douyin_aweme_comment`（字段常用：`comment_id`、`aweme_id`、`parent_comment_id`、`content`、`create_time`、`like_count`、`sub_comment_count`、`user_id`、`nickname`）。
  - 清洗后的 JSON（默认路径：`data/processed/douyin_comments_processed.json`，或由 `BaseAnalyzer` 的 `PathManager` 生成）。
- **输出目录**：
  - 结果：`data/results/`
  - 报告：`data/reports/`
  - 可视化：`data/visualizations/`
- **可视化策略**：不弹窗（无 `plt.show()`），仅保存 PNG。传入 `--no-viz` 可禁用图表生成。

---

### 二、模块算法详解

#### 1. 数据清洗 `modules/data_cleaning_optimized.py`
- 目标：统一数据准备流程，提升文本质量，为后续模块（时间/点赞/情感/相似度）提供高质量输入。
- 关键步骤：
  - 数据加载：支持数据库或清洗文件；支持 `--video-id`、`--limit`。
  - 质量检查：统计空内容、短内容、重复内容、缺失用户/视频ID。
  - 垃圾评论过滤：
    - 短内容（<5字符）；
    - 特殊字符比例 > 0.6；
    - 广告关键词（如“加微信/推广/贷款”等）；
    - 重复字符过多；
    - 可疑用户签名（过长或疑似联系方式）。
  - 文本清洗：去 HTML、URL、[]内容、空白；保留中/英/数字/中文标点。
  - 分词与停用词：`jieba.lcut` 后过滤停用词与过短词；统计 `word_count`。
  - 产出：清洗数据 JSON 保存到 `data/processed/`（由 `PathManager` 决定具体文件名）。
- 可视化：
  - 数据质量问题分布（条形图，自动跳过全 0 情况）；
  - 清洗流程统计（阶段计数折线图）；
  - 内容长度分布/词数分布（直方图，空数据友好处理）。
- 健壮性：大量 `.get()` 与空数据保护，失败时打印错误并关闭所有图表句柄。

#### 2. 点赞分析（优化版）`modules/like_analysis_optimized.py`
- 目标：分析点赞分布、父/子评论相关性、识别意见领袖、社会认同信号、跟随速度。
- 指标与方法：
  - 点赞分布统计：均值/中位数/标准差/Q25/Q75，与分段计数（0、1-4、5-19、20-99、100-499、500+）。
  - 父评论点赞范围影响：将父评论点赞映射到子评论，分档统计子评论点赞均值与相关性。
  - 意见领袖识别：父评论点赞阈值（默认≥20）、子评论量（≥3）、快速跟随比例（≥30%），生成影响力 Top10。
  - 社会认同信号：父评论点赞的高/低档对子评论点赞的影响与相关性。
  - 跟随速度：按阈值分段 immediate/quick/medium/slow/delayed 的数量与比例。
- 可视化：
  - 点赞分布（≤100 的直方图，含均值/中位数线）；
  - 点赞档位饼图；
  - 父/子评论点赞散点；
  - 意见领袖影响力/社会认同信号柱状图。
- 输出：结果与报告 JSON/CSV，图表 PNG 保存至对应目录。

#### 3. 情感分析（优化版）`modules/sentiment_analyzer_optimized.py`
- 目标：支持本地词典与阿里云 API 两种方式；统一统计、报告、可视化。
- 本地词典法：
  - 词典得分、否定词反转、程度副词加权；
  - 平均得分作为文本情感分数，极性阈值：≥0.3 正向，≤-0.3 负向，否则中性；
  - 置信度：分数绝对值映射，最大 0.9；
  - 输出字段：`sentiment`、`score`、`confidence`。
- 阿里云 API：
  - 首选 `aliyun-python-sdk-core` 调用 `GetSaChGeneral`（`alinlp`）；失败回退 HTTP；
  - 解析 `Data.result`，映射 `positive/negative/neutral`，并据概率生成 `score/confidence`；
  - 环境变量：`NLP_AK_ENV`、`NLP_SK_ENV`、`NLP_REGION_ENV`（默认 `cn-hangzhou`）。
- 统一输出：统计汇总、结果 CSV/JSON、报告 JSON（含分布、Top 正/负文本），可视化 4 图（分布饼图、分数/置信度直方、分数-置信度散点）。
- 可视化：加入空数据与 NaN 保护；仅保存不显示。

#### 4. 文本相似度（优化版）`modules/similarity_analysis_optimized.py`
- 目标：识别“模仿性评论”，量化从众心理的语义相似与时间接近性。
- 向量化：
  - 优先阿里云词向量 `GetWeChGeneral`（`alinlp`），参数：`Text`、`Size=100`、`Type=word`、`Operation=average`；
  - 解析 `Data.result` 中 `vec`，或多词向量取均值；
  - 调用失败自动降级至本地 TF-IDF（最大特征 100，1-2gram），并补零至 100 维。
- 相似度与配对：
  - 相似度：`cosine_similarity`；
  - 评论配对：出于性能与数据一致性，采用“相邻评论对 (i, i+1)”策略；
  - 时间差：取绝对秒差；
  - 判定规则：`similarity > similarity_threshold` 且 `time_diff < time_diff_threshold` 记为“模仿性”。默认阈值：0.7 与 3600s。
- 统计与输出：
  - 统计总评论、配对数、模仿性配对数/比例、相似度均值/中位数、平均时间差；
  - 导出相似度对、模仿性评论 CSV/JSON；
  - 可视化 4 图：相似度直方、时间差直方、相似度-时间差散点（阈值线）、模仿性比例饼图（空数据保护）。
- 环境变量：与情感分析相同（`NLP_AK_ENV`、`NLP_SK_ENV`）。

---

### 三、统一入口 `text_analysis_unified.py` 完整参数说明

#### 通用说明
- 运行格式：`python text_analysis/text_analysis_unified.py <module> [options]`
- 测试模式：任意子命令带 `--test` 时，如未指定 `--limit`，自动设置为 10。
- 可视化：所有模块默认“仅保存，不弹窗”；传入 `--no-viz` 可禁用图表生成。

#### cleaning（数据清洗）
- `--video-id` 字符串：仅清洗指定视频的评论；不传则全量。
- `--limit` 整数：限制清洗的记录数。
- `--test` 开关：测试模式（默认限制为 10 条）。
- `--no-save` 开关：不保存结果。
- `--no-report` 开关：不生成报告。
- `--no-viz` 开关：不生成可视化。

#### time（从众心理时间分析）
- `--use-cleaned-data` 开关：使用清洗数据（推荐）。
- `--video-id` 字符串：仅分析指定视频；不传则全量。
- `--limit` 整数：限制分析记录数。
- `--cleaned-data-path` 字符串：清洗数据文件路径（默认 `data/processed/douyin_comments_processed.json`）。
- `--test`、`--no-save`、`--no-report`、`--no-viz`：同上。

#### like（从众心理点赞分析）
- `--use-cleaned-data` 开关：使用清洗数据（推荐）。
- `--video-id`、`--limit`、`--cleaned-data-path`：同上。
- `--test`、`--no-save`、`--no-report`、`--no-viz`：同上。

#### sentiment（情感分析）
- `--use-cleaned-data` 开关：使用清洗数据（推荐）。
- `--type` 选择：`local`（本地词典）或 `aliyun`（默认）。
- `--video-id`、`--limit`、`--cleaned-data-path`：同上。
- `--test`、`--no-save`、`--no-report`、`--no-viz`：同上。
- 需配置环境变量（当 `--type aliyun`）：`NLP_AK_ENV`、`NLP_SK_ENV`、`NLP_REGION_ENV`(可选)。

#### similarity（文本相似度分析）
- `--use-cleaned-data` 开关：使用清洗数据（推荐）。
- `--video-id`、`--limit`、`--cleaned-data-path`：同上。
- `--similarity-threshold` 浮点：相似度阈值（默认 0.7）。
- `--time-diff-threshold` 整数：时间差阈值秒（默认 3600）。
- `--test`、`--no-save`、`--no-report`、`--no-viz`：同上。
- 需配置环境变量：`NLP_AK_ENV`、`NLP_SK_ENV`。

---

### 四、推荐分析流程（示例）
- 清洗（可选 `--video-id`）：
  - `python text_analysis/text_analysis_unified.py cleaning --video-id 123456`
- 时间分析（基于清洗数据）：
  - `python text_analysis/text_analysis_unified.py time --use-cleaned-data --video-id 123456`
- 点赞分析（基于清洗数据）：
  - `python text_analysis/text_analysis_unified.py like --use-cleaned-data --video-id 123456`
- 情感分析（阿里云，基于清洗数据）：
  - `python text_analysis/text_analysis_unified.py sentiment --use-cleaned-data --type aliyun --video-id 123456`
- 相似度分析（基于清洗数据，自定义阈值）：
  - `python text_analysis/text_analysis_unified.py similarity --use-cleaned-data --video-id 123456 --similarity-threshold 0.75 --time-diff-threshold 5400`

---

### 五、依赖与环境
- Python 包：`pandas`、`numpy`、`jieba`、`scikit-learn`、`matplotlib`、`seaborn`、`aliyun-python-sdk-core`、`python-dotenv`(可选)。
- 阿里云环境变量：
  - `NLP_AK_ENV`：AccessKey ID
  - `NLP_SK_ENV`：AccessKey Secret
  - `NLP_REGION_ENV`：Region，可选，默认 `cn-hangzhou`

---

### 六、可视化与报告约定
- 不弹窗：所有模块统一移除 `plt.show()`，保存后 `plt.close()`。
- 空数据保护：饼图/直方图/散点均在空数据或全 0 情况下给出友好提示，不抛异常。
- 输出命名：均包含时间戳，避免覆盖历史结果。
- 关闭可视化：添加 `--no-viz`。

---

### 七、设计取舍与已知限制
- 相似度配对采用“相邻评论对 (i, i+1)”以兼顾性能与数据一致性（避免依赖数据库中可能缺失的父子关系列）。如需严格父子关系分析，可在清洗数据中提供稳定的 `parent_comment_id` 并扩展配对逻辑。
- 阿里云 API 不可用时自动回退 TF-IDF，保证功能可用但精度可能下降。
- 清洗策略与阈值（长度、广告关键词、特殊字符比例等）可按业务调整。

---

如需我将该文档内容同步为英文版/增补更多图例，请告知。


