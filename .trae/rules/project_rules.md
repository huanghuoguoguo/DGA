# Trae AI IDE 规则集（用于代码生成与项目结构管理）

## 一、通用原则
1. 所有生成动作遵循“原地更新”原则：新代码直接替换旧代码，禁止出现 `*_enhanced.py`、`*_optimized.py` 等冗余文件。
2. 禁止在文件名、类名、函数名及注释中使用 `enhanced / unified / optimized / augmented` 等模糊词汇；如需升级，直接覆盖原实现。
3. 项目根目录保持极简，非必要不新建文件；临时文件一律写入 `.tmp/` 目录，并在任务结束后自动清理。
4. 简单测试脚本（如 `test_*.py`）通过验证后立即删除，禁止提交到版本库。
5. 不要主动生成 `result_*.md / summary_*.md` 等说明文档；仅在我显式指令下输出。

## 二、数据集规范
1. 格式统一：同一项目仅允许一种数据格式（`parquet` 优先，其次 `arrow`，最后 `jsonl`），禁止混用。
2. 目录结构：
   dataset/
   ├── raw/          # 原始数据，只读
   ├── interim/      # 中间缓存，可随时清空
   └── processed/    # 最终可用数据集，含：
       ├── train.parquet
       ├── val.parquet
       └── test.parquet
3. 划分比例固定：训练 80 % 验证 10 % 测试 10 %；函数内部禁止硬编码，通过 `split_ratio=(0.8,0.1,0.1)` 参数传入。
4. 数据加载入口唯一：所有读取必须通过 `dataset/load.py` 中的 `load_dataset(split: str)` 函数，禁止在业务代码中重复实现。

## 三、代码生成细则
1. 单文件职责清晰：一个 `.py` 文件只干一件事（`model.py / data.py / train.py`），禁止出现“瑞士军刀”式脚本。
2. 函数级替换：升级功能时，保持函数签名不变，仅修改函数体；若必须变更签名，提供一次性 `migrate.py` 脚本并在执行后删除。
3. 日志与输出：
   - 日志统一使用 `logging`，配置写入 `logging.yaml`，禁止随处 `print`
   - 控制台仅输出关键指标（`loss / acc / f1`），每 epoch 一行，禁止刷屏
4. 依赖管理：所有第三方库必须在 `requirements.txt` 中锁定版本；生成新依赖时同步写入，不留“TODO 安装”注释。
5. 配置集中：超参、路径、模型选型全部放入 `config.yaml`；禁止在代码里出现魔法数字或路径字符串。

## 四、文档与输出
1. 文档目录固定为 `docs/`，生成的图表、说明、API 文档一律写入该目录；根目录不得出现 `*.md` 或 `*.png`。
2. 不在保存时自动打开浏览器预览；不生成“恭喜完成”类 HTML 报告。
3. 仅在我指令 `&gt; generate report` 时才输出结构化结果（markdown/pdf），且默认存放位置 `docs/reports/`。

## 五、临时文件与清理
1. 临时文件写入 `.tmp/`（自动创建，已加入 `.gitignore`）。
2. 每次运行结束前执行清理钩子：
   ```python
   import shutil, os
   if os.path.exists('.tmp'): shutil.rmtree('.tmp')
Jupyter 缓存（.ipynb_checkpoints）、Python 缓存（__pycache__）、IDE 索引（.idea、.vscode）均在 .gitignore 中预设，生成项目时自动写入。
##  六、测试与验证
快速验证脚本统一命名 quick_test.py，放置项目根目录；验证通过即自动删除。
单元测试放在 tests/ 目录，使用 pytest；生成测试时同步写入 pytest.ini 配置，禁止冗余 if __name__ == "__main__" 块。
不在控制台打印详细断言信息；失败时仅输出一行 FAIL: <test_name> - see tests/log.txt。
## 七、版本提交辅助
生成代码后自动执行：
black . && isort . 格式化
flake8 --max-line-length=88 静态检查零警告后方可提交
提交信息模板固定：git commit -m "feat/fix: <brief description>"，禁止默认空白信息。
## 八、用户自定义规则（已融合优化）
不要使用增强、unified、optimized 等字眼	文件/函数/变量命名统一采用 <module>_v2.py、<function>_update() 方式，直接覆盖旧实现
数据集格式统一	仅保留一种格式（parquet），加载入口唯一，load_dataset(split='train')
数据集划分固定	代码内强制参数 split_ratio=(0.8,0.1,0.1)，不允许硬编码
简单测试文件验证后即删除	统一命名 quick_test.py，验证结束自动 os.remove(__file__)
项目保持干净	临时目录 .tmp/ + 自动清理钩子；缓存目录已加入 .gitignore
文档输出到 docs/	生成任何文档前检查 docs/ 存在，否则创建；禁止根目录散落 *.md
不主动输出结果 md	仅响应显式指令 > generate report，默认路径 docs/reports/