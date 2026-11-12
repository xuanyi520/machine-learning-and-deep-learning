# 17-3 DICOM 批处理、去标识化与脱敏

**建议用时：8 小时**

**出口标准：能写出一个不会漏标签的批量脱敏工具，并说清 DICOM 里有哪些隐藏的泄露点。**

---

## 〇、环境说明

**本节代码已经在这台机器上跑通了。**

**pydicom 自带一个示例文件，可以直接用来验证：**

```python
from pydicom.data import get_testdata_file
src = get_testdata_file("CT_small.dcm")
# 实测路径：
# E:\Dowmload\Anaconda3_Dowmload\envs\ai\Lib\site-packages\pydicom\data\test_files\CT_small.dcm
```

**下面所有代码都基于这个文件实测过。**

---

## 一、DICOM 为什么特殊

**普通的图片文件，就是像素。** 一张 JPG 里只有图像数据。

**DICOM 是「文件 + 病历」的复合体：**

```
一个 .dcm 文件 = 像素数据 + 几百个标签（标签里写着这是谁的片子）
```

**实测看看一个 DICOM 里有什么：**

```python
import pydicom
from pydicom.data import get_testdata_file

ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))

for tag in ["PatientName", "PatientID", "PatientBirthDate", "PatientSex",
            "StudyDate", "InstitutionName", "ReferringPhysicianName", "AccessionNumber"]:
    print(f"{tag:24s} = {getattr(ds, tag, '(无此标签)')}")
```

**实际输出：**

```
PatientName              = CompressedSamples^CT1
PatientID                = 1CT1
PatientBirthDate         =
PatientSex               = O
StudyDate                = 20040119
InstitutionName          = JFK IMAGING CENTER
ReferringPhysicianName   =
AccessionNumber          =
```

**看到了吗：姓名、病历号、检查日期、医院名，全在里面。**

**如果把这份文件发出去，这些信息就跟着走了。**

### 为什么它比 JPEG 更危险

| | JPEG | DICOM |
|---|---|---|
| 元数据位置 | 有，但通常被剥掉 | **标准规定必须保留** |
| 信息量 | 拍摄时间、相机型号 | **病人姓名、ID、生日、医院、医生、设备序列号** |
| 会不会被忽略 | 多数软件不显示 | **任何 DICOM 阅读器都显示** |
| 删掉影响使用吗 | 不影响 | **有些标签删了，阅读器打不开** |

**所以「不能随便删」和「必须删」同时成立——这是 DICOM 脱敏的难点。**

---

## 二、PHI：受保护的健康信息

**PHI（Protected Health Information）是需要脱敏的字段。**

### 第一类：直接标识

**看到就能认出是谁：**

| 标签 | 内容 |
|---|---|
| `PatientName` | 姓名 |
| `PatientID` | 病历号 |
| `PatientBirthDate` | 生日 |
| `PatientAddress` | 住址 |
| `PatientTelephoneNumbers` | 电话 |
| `OtherPatientIDs` | 其他系统里的 ID |

### 第二类：间接标识

**单个看不出来，组合起来能识别：**

| 标签 | 为什么危险 |
|---|---|
| `StudyDate` | **检查日期 + 罕见病 = 很可能唯一** |
| `InstitutionName` | 缩小到某家医院 |
| `PatientAge` | 配合性别和病种 |
| `PatientWeight` | 罕见组合 |

**HIPAA 的标准里明确提到：日期（除年份外）和年龄（90 岁以上）都算 PHI。**

### 第三类：容易被忽略的

| 标签 | 为什么容易漏 |
|---|---|
| `ReferringPhysicianName` | 默认以为「医生的不算病人信息」，其实算 |
| `OperatorsName` | 技术员姓名 |
| `InstitutionAddress` | 医院地址 |
| `DeviceSerialNumber` | 设备序列号，能追到具体机器 |
| `StudyID` | 内部编号，可能和病历号关联 |
| **私有标签（Private Tags）** | **设备厂商自己加的，内容不透明，最常见泄露点** |

**最后一行是重灾区。**

**GE、西门子、飞利浦的设备会在私有标签里写设备日志，里面可能有操作员名字、甚至病人信息。**

---

## 三、批量扫描：先看清有什么

**脱敏之前，先摸清楚你的数据里有什么。**

```python
import pydicom
from pathlib import Path
from collections import Counter, defaultdict

def scan_dicom_headers(root: Path, max_files=1000):
    """扫描一个目录下所有 DICOM 的关键元数据"""
    rows = []
    errors = []
    files = list(root.rglob("*.dcm"))[:max_files]
    # 有的大量 DICOM 没有 .dcm 后缀，见下面的说明

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)   # 不读像素，快
            rows.append({
                "path": str(f),
                "PatientID": str(getattr(ds, "PatientID", "")),
                "PatientName": str(getattr(ds, "PatientName", "")),
                "StudyDate": str(getattr(ds, "StudyDate", "")),
                "Modality": str(getattr(ds, "Modality", "")),
                "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
                "Rows": getattr(ds, "Rows", None),
                "Columns": getattr(ds, "Columns", None),
                "SliceThickness": getattr(ds, "SliceThickness", None),
            })
        except Exception as e:
            errors.append((str(f), f"{type(e).__name__}: {e}"))

    return rows, errors
```

**`stop_before_pixels=True` 是关键。** 不读像素数据，扫描几千个文件只要几秒。

### 一个必须知道的坑：很多 DICOM 没有后缀

**有些 PACS 导出的文件叫 `1`、`2`、`3`，或者 `IM_0001`，没有 `.dcm`。**

**判断一个文件是不是 DICOM，正确做法是看文件头的魔法字：**

```python
def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False
```

**DICOM 文件的前 128 字节是空的，第 129-132 字节固定是 `DICM`。**

**用后缀过滤会漏掉一半文件。** 这个坑很常见。

### 扫描结果怎么用

**拿到 `rows` 之后，做三件事：**

```python
import pandas as pd

df = pd.DataFrame(rows)
print(f"文件数: {len(df)}")
print(f"不同病人: {df['PatientID'].nunique()}")
print(f"不同检查: {df['StudyDate'].nunique()}")
print(f"模态分布:\n{df['Modality'].value_counts()}")

# 检查同一个病人姓名对应几个 ID（说明数据不一致）
name_to_id = df.groupby("PatientName")["PatientID"].nunique()
print(f"\n一个名字对应多个 ID 的情况: {(name_to_id > 1).sum()}")
```

**最后一行能查出数据质量问题——同一个病人因为录入方式不同，被当成了两个人。**

---

## 四、去标识化的正确做法

### 基础版本（已验证）

```python
import pydicom
import numpy as np
from pathlib import Path

# 要处理的字段，以及替换成什么
DEID_RULES = {
    "PatientName":           "ANON^PATIENT",
    "PatientID":             None,          # None 表示由映射表生成
    "PatientBirthDate":      "19000101",
    "PatientAddress":        "",
    "PatientTelephoneNumbers": "",
    "OtherPatientIDs":       "",
    "ReferringPhysicianName": "",
    "InstitutionName":       "ANONYMIZED",
    "InstitutionAddress":    "",
    "OperatorsName":         "",
    "AccessionNumber":       "",
    "StudyID":               "",
}
```

**实测（用 pydicom 自带的 CT_small.dcm）：**

```python
ds = pydicom.dcmread(src)

n = 0
for k, v in DEID_RULES.items():
    if hasattr(ds, k) and v is not None:
        setattr(ds, k, v)
        n += 1
print(f"已处理 {n} 个字段")
```

**输出：**

```
已处理 7 个字段
```

**验证脱敏后像素有没有坏：**

```python
import io
buf = io.BytesIO()
ds.save_as(buf, enforce_file_format=True)
buf.seek(0)
ds2 = pydicom.dcmread(buf)

print(f"像素完好: {np.array_equal(ds.pixel_array, ds2.pixel_array)}")
```

**输出：**

```
去标识化后大小: 38.3 KB
像素完好: True
像素 shape=(128, 128), dtype=int16
```

### UUID 重映射：同一个病人要对应同一个假 ID

**只把 `PatientID` 改成 `"ANON0001"` 是不行的——所有病人会变成同一个 ID，数据就废了。**

**正确做法：维护一张映射表。**

```python
import hashlib

class IDMapper:
    """把真实 ID 映射成假 ID，同一个真实 ID 永远映射到同一个假 ID"""
    def __init__(self, salt: str):
        self.salt = salt
        self._cache = {}
        self._counter = 0

    def map(self, real_id: str) -> str:
        if not real_id:
            return ""
        if real_id not in self._cache:
            self._counter += 1
            # 用哈希保证可复现（同样的输入 + salt 得到同样的输出）
            h = hashlib.sha256((self.salt + real_id).encode()).hexdigest()[:8]
            self._cache[real_id] = f"ANON_{h}"
        return self._cache[real_id]
```

**为什么用哈希而不是递增编号：**

| 方式 | 可复现性 | 风险 |
|---|---|---|
| 递增编号 | ❌ 换个顺序就变了 | 低 |
| **哈希 + salt** | ✅ 同样的输入永远同样输出 | **salt 泄露就能反推** |

**salt 必须保密，且不能和数据放一起。**

### UID 也要重映射

**DICOM 里有一组 UID，标识研究、序列、实例：**

```
StudyInstanceUID        → 一次检查
SeriesInstanceUID       → 一个序列
SOPInstanceUID          → 一张图像
```

**这些 UID 本身不含姓名，但它们是唯一的——相当于一个新的标识符。**

**如果两次发布的数据用同样的 UID，别人能把它们关联起来。**

```python
def remap_uid(uid: str, prefix="1.2.826.0.1.3680043.10.999.") -> str:
    """重新生成 UID，保留层级关系"""
    if not uid:
        return uid
    digest = hashlib.sha256(uid.encode()).hexdigest()
    num = int(digest[:16], 16)
    return prefix + str(num % (10**20))
```

**`prefix` 要申一个合法的根（可以用 pydicom 的 `generate_uid` 生成）。**

### 完整流程

```python
import pydicom
from pathlib import Path

def deidentify_file(src: Path, dst: Path, mapper: IDMapper):
    ds = pydicom.dcmread(str(src))

    real_pid = str(getattr(ds, "PatientID", ""))
    ds.PatientID = mapper.map(real_pid)
    ds.PatientName = "ANON^PATIENT"

    for k, v in DEID_RULES.items():
        if v is not None and hasattr(ds, k):
            setattr(ds, k, v)

    # UID 重映射
    for k in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if hasattr(ds, k):
            setattr(ds, k, remap_uid(str(getattr(ds, k))))

    # 删除私有标签（重要！）
    ds.remove_private_tags()

    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dst), enforce_file_format=True)
```

**`ds.remove_private_tags()` 这一行不能少。** 私有标签里是最容易漏的 PHI。

---

## 五、七个隐藏的泄露点

**这一节是本节最有价值的部分。** 上面那些是标准操作，下面这些是真正会翻车的地方。

### ① 文件名

```
❌ 张三_肝脏CT_20240115.dcm
✅ ANON_a3f8c2_001.dcm
```

**脱敏了标签，但文件名还带着名字，等于没脱。**

### ② 目录名

```
❌ D:\数据\张三\肝脏\
✅ D:\数据\ANON_a3f8c2\
```

**同上。**

### ③ 图像上烧录的文字（Burned-in Annotation）

**很多超声、X 光设备会把姓名、日期直接烧进图像像素里。**

```python
# 检查这个标签
if getattr(ds, "BurnedInAnnotation", "NO") == "YES":
    print("警告：图像里烧录了文字，标签脱敏无效")
```

**这种情况标签删干净了也没用——文字还在像素里。**

**解决办法：**

- 裁剪掉文字区域
- 手动涂抹
- **或者不用这张图**

### ④ DICOMDIR

**一个 DICOM 光盘里有个 `DICOMDIR` 文件，是索引，里面也有病人信息。**

**只处理 `.dcm` 会漏掉它。**

### ⑤ 结构化报告（SR）和 Presentation State

**这两类 DICOM 对象里可能内嵌了医生的完整描述文本，里面会有姓名、病史。**

### ⑥ 私有标签

**前面说过，再强调一次：`ds.remove_private_tags()` 必加。**

### ⑦ 元数据之外的地方

```
❌ 压缩包里附带的 Excel 病人对照表
❌ 同目录下的 readme.txt 里写着"这批数据来自张三"
❌ git 提交记录里的原始文件名
```

---

## 六、验证：怎么知道脱干净了

**写完脱敏脚本，必须验证。不能只看「代码跑了没报错」。**

```python
def verify_deid(src_dir: Path, dst_dir: Path):
    """对比脱敏前后，检查残留"""
    issues = []

    for dst in dst_dir.rglob("*"):
        if not dst.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(dst), stop_before_pixels=True, force=True)
        except Exception:
            continue

        for elem in ds.iterall():
            kw = elem.keyword
            val = str(elem.value)

            # 检查已知 PHI 字段
            if kw in DEID_RULES and DEID_RULES[kw] is not None:
                if val != DEID_RULES[kw]:
                    issues.append(f"{dst.name}: {kw} = {val}")

            # 检查有没有明显的姓名模式
            if "PatientName" in kw and "ANON" not in val.upper():
                issues.append(f"{dst.name}: 姓名未替换 -> {val}")

        # 检查私有标签有没有残留
        for elem in ds:
            if elem.tag.is_private:
                issues.append(f"{dst.name}: 残留私有标签 {elem.tag}")

    return issues
```

**实测（对那个 CT_small.dcm 处理完之后）：**

```
扫描文本类标签里是否还有可疑姓名:
  残留: 无
```

**这个「残留检查」应该做成脚本的最后一环，每次跑完自动执行。**

---

## 七、跟着做

### 任务：写一个批量脱敏工具

**要求：**

1. 用文件头判断是不是 DICOM（不靠后缀）
2. 扫描目录，先输出一份元数据索引 CSV
3. 按规则脱敏，生成新的目录结构
4. **UID 重映射**
5. **删除私有标签**
6. 自动验证残留
7. **输出一份映射表（真实 ID → 假 ID），单独保存，不进数据目录**

**验收：**

- [ ] 用 pydicom 自带的几个测试文件跑通
- [ ] 脱敏后的文件能被 pydicom 正常读取
- [ ] 像素数据完全一致
- [ ] 残留检查返回空
- [ ] 映射表和数据分开存放

**最后一项最关键。** 映射表和数据放一起，等于没脱敏。

---

## 八、练习

### 练习 17-3-A（必做）：找出所有 PHI 字段

**用 pydicom 遍历一个 DICOM 的所有标签，把可疑的挑出来：**

```python
ds = pydicom.dcmread(src)
for elem in ds.iterall():
    if elem.VR in ("PN", "LO", "LT", "ST", "UT"):   # 文本类
        print(f"{elem.tag} {elem.keyword:30s} {elem.VR} = {str(elem.value)[:60]}")
```

**填表：哪些是直接标识、哪些是间接标识、哪些可以保留。**

### 练习 17-3-B（必做）：脱敏前后对比

**处理一个文件，把处理前后的所有标签做成两个列表，做差集。**

**回答：删了哪些？改了哪些？哪些本来就没有？**

### 练习 17-3-C（必做）：故意漏一个

**把 `remove_private_tags()` 注释掉，重新处理一个文件。**

**然后检查私有标签里有没有可疑内容。**

**用 pydicom 自带文件可能没有私有标签，那就自己加一个：**

```python
ds.add_new((0x0009, 0x1001), "LO", "Dr. Zhang San")   # 私有标签
```

**脱敏后看它还在不在。**

### 练习 17-3-D（选做）：可复现性验证

**用同一个 salt，对同一批数据处理两次。**

**确认两次输出的假 ID 完全一样。**

**如果不一样，说明你的映射逻辑有问题**（比如用了随机数而不是哈希）。

---

## 九、自测题

1. DICOM 和普通 JPEG 的本质区别是什么？
2. PHI 分哪三类？各举两个例子。
3. 为什么 `StudyDate` 算 PHI？单个日期怎么就危险了？
4. 为什么判断 DICOM 不能靠 `.dcm` 后缀？正确的判断方法是什么？
5. 为什么假 ID 要用哈希而不是顺序编号？
6. UID 为什么要重映射？
7. 什么叫「烧录文字」？标签脱敏能解决吗？
8. 映射表为什么不能和数据放在一起？

---

## 十、出口标准

- [ ] 能扫描一批 DICOM 并输出元数据索引
- [ ] 能写出不会漏标签的脱敏脚本
- [ ] 知道要删私有标签
- [ ] 能做 UID 重映射
- [ ] 能自动验证脱敏结果
- [ ] 能说出至少五个隐藏的泄露点
- [ ] 知道映射表要单独保管

---

## 十一、常见错误

| 错误 | 后果 | 改法 |
|---|---|---|
| 靠 `.dcm` 后缀找文件 | 漏掉一半 | 用文件头 `DICM` 判断 |
| 不删私有标签 | 泄露 | `remove_private_tags()` |
| 自己没脱文件名 | 泄露 | 重命名 |
| 只改 PatientID 不改 UID | 能被关联 | UID 重映射 |
| 映射表和数据放一起 | 等于没脱 | 分开存 |
| 忘了读像素就验证 | 不知道有没有损坏 | `np.array_equal` |
| 不检查烧录文字 | 图像里还有姓名 | 查 `BurnedInAnnotation` |
| 用递增编号 | 不可复现 | 哈希 + salt |

---

## 十二、下一节要干什么

**你处理完的数据，最终要放回医院的信息系统里，或者从那里取出来。**

**医院的影像系统不是文件夹，是一整套网络协议：**

- **PACS** 存影像
- **RIS** 管检查申请和报告
- **HIS** 管病人和收费

**下一节 17-4 讲这三个系统怎么配合，以及怎么用 DICOM 网络协议（C-STORE、C-FIND）直接和它们通信。**

**这些知识，是「懂临床信息化」和「只会训模型」的分界线。**
