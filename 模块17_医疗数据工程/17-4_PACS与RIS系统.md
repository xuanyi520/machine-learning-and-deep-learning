# 17-4 PACS / RIS / DICOM 网络

**建议用时：6 小时**

**出口标准：能说清 PACS、RIS、HIS 各自管什么，能用 pynetdicom 做一次 C-ECHO 和 C-FIND。**

---

## 〇、环境说明

**这一节需要装：**

```
E:\Dowmload\Anaconda3_Dowmload\envs\ai\python.exe -m pip install pynetdicom
```

**约 5 MB，纯 Python。**

**本节代码我还没实跑（包没装）。但有了它，你可以在本机起一个「假 PACS」，自己和自己通信。**

---

## 一、三个系统，各管什么

**医院信息系统的缩写满天飞，先把最基础的三层理清楚。**

```
┌─────────────────────────────────────────────┐
│  HIS（医院信息系统）                          │
│  挂号、收费、住院、药房、医保                  │
│  核心实体：病人（Patient）                     │
└──────────────────┬──────────────────────────┘
                   │ 医嘱：给某人开一个"腹部CT平扫"
                   ▼
┌─────────────────────────────────────────────┐
│  RIS（放射信息系统）                          │
│  检查申请、预约、登记、报告书写、审核          │
│  核心实体：检查（Study）                       │
└──────────────────┬──────────────────────────┘
                   │ 检查到了，派给某台设备
                   ▼
┌─────────────────────────────────────────────┐
│  PACS（影像归档与通信系统）                    │
│  存图像、调阅图像、分发图像                    │
│  核心实体：序列（Series）、图像（Image）        │
└─────────────────────────────────────────────┘
```

### 一句话区分

| 系统 | 一句话 | 类比 |
|---|---|---|
| **HIS** | 管「谁来了」 | 医院的 ERP |
| **RIS** | 管「做了什么检查、报告写了什么」 | 体检中心的前台系统 |
| **PACS** | 管「片子存在哪」 | 影像的网盘 |

**你在影像 AI 项目里对接的，绝大多数情况是 PACS。**

### 实际的检查流程

```
① 病人挂号                        → HIS 生成 PatientID
② 医生开单「腹部CT平扫」           → RIS 生成 AccessionNumber
③ 病人到放射科登记                → RIS 生成 StudyInstanceUID
④ 技师在 CT 机上扫描              → 图像写进 PACS，带 SeriesInstanceUID
⑤ 图像传到医生工作站              → 从 PACS 调阅
⑥ 医生写报告                      → RIS 里保存
⑦ 报告审核发布                    → 回到 HIS
```

**你要拿数据，通常是在第 ④ 步之后，从 PACS 里取。**

---

## 二、DICOM 网络协议

**PACS 不是「共享文件夹」，它用一套专门的网络协议通信。**

**这套协议叫 DIMSE（DICOM Message Service Element），有四个核心操作：**

| 操作 | 干什么 | 类比 |
|---|---|---|
| **C-ECHO** | 检查对方在不在 | ping |
| **C-FIND** | 查询，只拿元数据不拿图像 | 搜索 |
| **C-MOVE** | 让 PACS 把图像发到某个地方 | 下载（但由对方发起） |
| **C-STORE** | 把图像推送给对方 | 上传 |

### 一个反直觉的点：C-MOVE 不是你「拉」，是让对方「推」

```
你想下载图像：
  你的程序 ──C-MOVE(发到"目的地A")──> PACS
  PACS ──C-STORE(把图像推过来)──> 目的地A

所以你要先起一个 C-STORE 服务端，等着接。
```

**这是新手最容易懵的地方。**

### AE Title：DICOM 世界的「主机名」

**每个参与方都要有一个 AE Title（Application Entity Title），最多 16 个字符。**

```
你的程序：  MY_AI_CLIENT
PACS：      PACS_SERVER
```

**C-MOVE 时要告诉 PACS「发到哪个 AE」，PACS 会在它的配置表里查这个 AE 对应的 IP 和端口。**

**所以对接 PACS，通常需要对方把你加进白名单。**

---

## 三、本机跑一个假 PACS

**不用连医院的系统，自己和自己通信就行。**

```python
# demo_pacs.py
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import Verification, PatientRootQueryRetrieveInformationModelFind
import pydicom
from pydicom.dataset import Dataset

AE_TITLE = "DEMO_PACS"

def handle_echo(event):
    print("收到 C-ECHO")
    return 0x0000                    # 成功

def handle_find(event):
    print("收到 C-FIND")
    for i in range(3):
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.PatientID = f"ANON{i:04d}"
        ds.StudyInstanceUID = f"1.2.3.4.{i}"
        ds.StudyDate = "20260101"
        yield (0xFF00, ds)           # 0xFF00 = 还有更多
    yield (0x0000, None)             # 结束

handlers = [
    (evt.EVT_C_ECHO, handle_echo),
    (evt.EVT_C_FIND, handle_find),
]

ae = AE(ae_title=AE_TITLE)
ae.add_supported_context(Verification)
ae.add_supported_context(PatientRootQueryRetrieveInformationModelFind)

print(f"假 PACS 启动，AE Title = {AE_TITLE}, 端口 11112")
ae.start_server(("127.0.0.1", 11112), block=True, evt_handlers=handlers)
```

**另开一个终端跑客户端：**

```python
# demo_client.py
from pynetdicom import AE
from pynetdicom.sop_class import Verification, PatientRootQueryRetrieveInformationModelFind
from pydicom.dataset import Dataset

ae = AE(ae_title="MY_AI_CLIENT")
ae.add_requested_context(Verification)
ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)

assoc = ae.associate("127.0.0.1", 11112, ae_title="DEMO_PACS")

if assoc.is_established:
    # 1. C-ECHO
    status = assoc.send_c_echo()
    print(f"C-ECHO 状态: 0x{status.Status:04X}")

    # 2. C-FIND
    ds = Dataset()
    ds.QueryRetrieveLevel = "STUDY"
    ds.PatientID = ""
    ds.StudyInstanceUID = ""
    ds.StudyDate = ""

    for status, identifier in assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind):
        if status and status.Status in (0xFF00, 0xFF01):
            print(f"  找到: {identifier.PatientID}  {identifier.StudyDate}")
        elif status:
            print(f"  结束: 0x{status.Status:04X}")

    assoc.release()
else:
    print("连接失败")
```

**这段话解释了 DICOM 通信的核心：**

```
① 建立关联（associate）—— 双方协商支持哪些操作
② 发请求，等响应
③ 断开关联
```

---

## 四、真实对接 PACS 的流程

**上面是本机演示。真去医院，流程是这样：**

### 第一步：拿到网络信息

| 你需要 | 对方给你 |
|---|---|
| PACS 的 IP | ✅ |
| PACS 的端口 | ✅（通常是 104、11112） |
| PACS 的 AE Title | ✅ |
| **你的 AE Title** | 你要报给人家，他们加白名单 |

### 第二步：明确「你有权做什么」

**四种权限，要分开申请：**

```
□ C-ECHO   测试连通     → 一般都会给
□ C-FIND   查元数据     → 通常给，但可能限制查询范围
□ C-MOVE   取图像       → 最敏感，要说明用途
□ C-STORE  上传图像     → 一般不给（会污染 PACS）
```

**⚠️ 大多数医院不允许第三方程序往 PACS 里写（C-STORE）。**

**因为写了就删不掉，而且会影响医生的调阅。**

### 第三步：先测连通性

```python
assoc = ae.associate(pacs_ip, pacs_port, ae_title=pacs_ae)
if assoc.is_established:
    status = assoc.send_c_echo()
    print("通了" if status.Status == 0x0000 else f"失败 0x{status.Status:04X}")
    assoc.release()
```

**C-ECHO 都过不了，别往下做。**

### 第四步：查数据范围

```python
# 先查有多少个检查
ds = Dataset()
ds.QueryRetrieveLevel = "STUDY"
ds.StudyDate = "20260101-20260131"     # 日期范围
ds.StudyInstanceUID = ""
ds.PatientID = ""

count = 0
for status, ident in assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind):
    if status and status.Status in (0xFF00, 0xFF01):
        count += 1
print(f"这个月有 {count} 个检查")
```

**先查数量再决定取多少。别一上来就 C-MOVE 几万张图。**

### 第五步：取图像

```python
from pynetdicom import AE, evt, StoragePresentationContexts
import pydicom
from pathlib import Path

OUT = Path(r"F:\AI学习环境\pacs_data")
OUT.mkdir(parents=True, exist_ok=True)

def handle_store(event):
    ds = event.dataset
    ds.file_meta = event.file_meta
    path = OUT / f"{ds.SOPInstanceUID}.dcm"
    ds.save_as(str(path), enforce_file_format=True)
    return 0x0000

handlers = [(evt.EVT_C_STORE, handle_store)]

# 起一个接收端
store_ae = AE(ae_title="MY_STORE_SCP")
store_ae.supported_contexts = StoragePresentationContexts
server = store_ae.start_server(("0.0.0.0", 11113), block=False, evt_handlers=handlers)

# 发 C-MOVE，让 PACS 推过来
mover = AE(ae_title="MY_AI_CLIENT")
mover.add_requested_context(PatientRootQueryRetrieveInformationModelMove)
assoc = mover.associate(pacs_ip, pacs_port, ae_title=pacs_ae)

ds = Dataset()
ds.QueryRetrieveLevel = "STUDY"
ds.StudyInstanceUID = "1.2.3.4.5"

for status, ident in assoc.send_c_move(ds, "MY_STORE_SCP", PatientRootQueryRetrieveInformationModelMove):
    print(f"状态 0x{status.Status:04X}")

assoc.release()
server.shutdown()
```

**关键：`send_c_move` 的第二个参数是「发到哪个 AE」，不是「发到哪个 IP」。**

**那个 AE 必须是 PACS 白名单里已知的。所以本机测试时，`MY_STORE_SCP` 的 IP 要能被 PACS 访问到。**

---

## 五、IHE：更上一层

**DICOM 只规定了「怎么传」，没规定「什么时候传、谁传给谁」。**

**IHE（Integrating the Healthcare Enterprise）定义了完整的业务场景。**

### 三个和你相关的 IHE 集成模式

| 模式 | 场景 |
|---|---|
| **Scheduled Workflow** | 从开单到出报告的完整流程 |
| **XDS** | 跨机构的文档共享 |
| **AIW-I**（AI Workflow for Imaging） | **AI 结果怎么回到临床** |

**最后一个和你最相关：AI 分析结果怎么让医生看到？**

**常见做法：**

```
① AI 结果写成 DICOM 对象（Segmentation Object 或 Structured Report）
② 用 C-STORE 推到 PACS
③ 医生在工作站上打开原图时，AI 结果作为「图层」叠加显示
```

**这就是「AI 集成进临床」的技术路径。**

**面试时能说出这一套，比会说「我用 Flask 部署了模型」高一个层级。**

---

## 六、跟着做

### 任务一：本机跑通 C-ECHO

**起一个假 PACS，用客户端连上去。**

**验收：**

- [ ] C-ECHO 返回 0x0000
- [ ] 服务端打印「收到 C-ECHO」

### 任务二：实现 C-FIND

**让假 PACS 返回 3 条虚构的检查记录，客户端打印出来。**

**验收：**

- [ ] 客户端能拿到 3 条
- [ ] 能正确处理 0xFF00（还有）和 0x0000（结束）

### 任务三：本机跑通 C-STORE

**客户端把一个 `.dcm` 文件推给假 PACS，服务端保存到磁盘。**

**用 pydicom 自带的所有测试文件：**

```python
from pydicom.data import get_testdata_files
for f in get_testdata_files("*.dcm")[:5]:
    print(f)
```

**验收：**

- [ ] 服务端能收到并保存
- [ ] 保存的文件能被 pydicom 读出来

---

## 七、练习

### 练习 17-4-A（必做）：AE Title 的长度限制

**AE Title 最长 16 个字符。** 试着传一个 17 个字符的，看报什么错。

**记录错误信息。**

### 练习 17-4-B（必做）：QueryRetrieveLevel 的层级

**DICOM 查询有四个层级：**

```
PATIENT → STUDY → SERIES → IMAGE
```

**分别用四个层级查同一个数据集，看返回的字段有什么不同。**

**填表：每个层级返回什么关键字段？**

### 练习 17-4-C（必做）：CT 和 MR 的区别

**用你的假 PACS，返回 Modality 为 CT 和 MR 的记录各两条。**

**客户端用 `Modality="CT"` 过滤，看能不能只拿到 CT 的。**

**这一题让你理解「查询条件是怎么工作的」。**

### 练习 17-4-D（选做）：画一张架构图

**用 mermaid 画出「病人从挂号到 AI 出结果」的完整数据流，标出每一步用的是哪个系统、哪个协议。**

**画完贴在笔记里。面试时能直接讲。**

---

## 八、自测题

1. PACS、RIS、HIS 各自管什么？用一个词概括。
2. C-ECHO、C-FIND、C-MOVE、C-STORE 分别是什么？对应网络操作里的什么？
3. 为什么 C-MOVE 需要你先起一个服务端？
4. AE Title 是什么？为什么需要白名单？
5. C-FIND 返回的 0xFF00 和 0x0000 分别什么意思？
6. 为什么医院一般不给第三方 C-STORE 权限？
7. IHE 在 DICOM 之上补了什么？
8. AI 结果怎么回到临床？用什么 DICOM 对象？

---

## 九、出口标准

- [ ] 能用一句话区分 PACS、RIS、HIS
- [ ] 能说清四个 DIMSE 操作
- [ ] 能本机跑通 C-ECHO 和 C-FIND
- [ ] 知道对接真实 PACS 要准备什么
- [ ] 知道 C-MOVE 的「推」机制
- [ ] 能说出 AI 结果回到临床的路径

---

## 十、常见错误

| 错误 | 现象 | 原因 | 改法 |
|---|---|---|---|
| AE Title 超过 16 字符 | 关联失败 | 协议限制 | 缩短 |
| 以为 C-MOVE 是「拉」 | 收不到图像 | 它是推 | 先起接收端 |
| 没加白名单 | 连不上或推不过去 | 对方不认识你 | 报自己的 AE 和 IP |
| 一次 C-MOVE 几万张 | 把 PACS 拖垮 | 没控制量 | 先 C-FIND 查数量 |
| 忘 `assoc.release()` | 连接堆积 | 没释放 | 用完就放 |
| 用 C-STORE 往 PACS 写 | 被拒或污染数据 | 权限 | 一般不要 |

---

## 十一、下一节要干什么

**现在你能拿到数据、能脱敏、能对接 PACS 了。**

**但一个项目做到第三个月，你面对的是一个有几千个文件、十几个版本的数据集。**

**「上次那个效果好的实验，用的是哪一版预处理？」——答不上来。**

**下一节 17-5 讲数据规范：目录怎么建、文件怎么命名、版本怎么管。**

**这一节不涉及算法，但它决定了你的项目能不能被别人接手。**
