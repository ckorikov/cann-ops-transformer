# 1. FA TilingKey占位参数说明

FlashAttentionScore的TilingKey的生成规则为，基于base值（1e19）组装的十进制数字，通过接口`GET_TILINGKEY()`获取。不同模板的占位个数不同，因此base值不同，TilingKey位号也不相同，但基于公共的分类方式。

## 1.1 FA TilingKey关键参数

最多支持19个参数，当前实现包含以下关键参数，从低位到高位，从右向左依次是：\
UB0、UB1、Block、ImplMode、DataType、Layout、Bmm1Format、Bmm2Source、Sparse、BigDoubleBuffer、HasDropOut、HasAttenMask、HasPse、EnableL1Reuse，各参数说明如下：
<br>

| TilingKey位号 | TilingKey位名称 | 作用 | 表示方式 | 取值范围 | 取值含义 | 涉及模板 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | UB0 | 表示UB核内切分的轴 | 枚举AxisEnum | 0-9 | 0：B <br> 1：N2 <br> 2：G <br> 3：S1 <br> 4：S2 <br> 5：D <br> 9：NONE | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 1 | UB1 | 同上，允许最多切分两根轴 | 枚举AxisEnum | 0-9 | 同上 | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 2 | Block | 核间切分轴 | 枚举AxisEnum | 0-9 | 同上 | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 3 | ImplMode | 是否使能高精度的无效行计算 | 枚举ImplModeEnum | 0-2 | 0：HIGH_PRECISION <br> 1：HIGH_PERFORMANCE <br> 2：INVALID_LINE_HIGH_PRECISION | TND <br> SameAB <br> S1S2 <br> S1 <br> B |
| 4 | DataType | 数据类型 | 枚举DtypeEnum | 0-3 | 0：FLOAT16 <br> 1：FLOAT32 <br> 2：BFLOAT16 <br> 3：FLOAT16_PRECISION | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 5 | Layout | 数据格式 | 枚举LayoutEnum | 0-3 | 0：BSH <br> 1：SBH <br> 2：BNSD <br> 3：TND | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 6 | Bmm1Format | Bmm1输出格式 | 枚举CubeFormatEnum | 0-1 | 0：ND <br> 1：NZ | SameAB <br> S1S2 <br> S1 |
| 7 | Bmm2Source | Bmm2数据来源 | 枚举CubeInputSourceEnum | 0-1 | 0：GM <br> 1：L1 | S1 |
| 8 | Sparse | sparse mode | 枚举SparseEnum | 0-9 | 0：ALL <br> 1：NONE <br> 2：ANY <br> 3：CAUSAL <br> 4：BAND <br> 5：PREFIX <br> 6：BAND_COMPRESS <br> 7：RIGHT_DOWN_CAUSAL <br> 8：RIGHT_DOWN_CAUSAL_BAND <br> 9：BAND_LEFT_UP_CAUSAL | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 9 | BigDoubleBuffer | 双缓冲机制 | 枚举PerformanceOrientedEnum | 1-2 | 1：BIG_BUFFER <br>  2：BIG_DOUBLE_BUFFER | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 10 | HasDropOut | 是否支持可选输入drop_mask | bool | 0-1 | 0：False <br> 1：True | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 11 | HasAttenMask | 是否支持可选输入atten_mask | bool | 0-1 | 0：False <br> 1：True | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 12 | HasPse | 是否支持可选输入real_shift | bool | 0-1 | 0：False <br> 1：True | TND <br> TNDSameAB <br> SameAB <br> S1S2 <br> S1 <br> B |
| 13 | EnableL1Reuse | 是否使能L1复用 | bool | 0-1 | 0：False <br> 1：True | S1S2 <br> S1 |

<br>

## 1.2 FA TilingKey补充参数

其余场景，可以依次在后面定义自己的位域和值。模板中还包含以下其他可以配置的参数，供参考。
<br>

| TilingKey位号 | TilingKey位名称 | 作用 | 表示方式 | 取值范围 | 取值含义 | 涉及模板 |
| --- | --- | --- | --- | --- | --- | --- |
| - | MatmulPolicyType | MatmulPolicy | 枚举MatmulPolicyType | 0-1 | 0：MATMUL_POLICY_NORMAL <br> 1：MATMUL_POLICY_UNSPLITK | SameAB |
| - | S1TemplateType | S1轴shape | 枚举STemplateType | 0-8 | 0：S_TEMPLATE_UNKNOW <br> 1：ALIGNED_16 <br> 2：ALIGNED_32 <br> 3：ALIGNED_48 <br> 4：ALIGNED_64 <br> 5：ALIGNED_80 <br> 6：ALIGNED_96 <br> 7：ALIGNED_112 <br> 8：ALIGNED_128 | S1 <br> B |
| - | S2TemplateType | S2轴shape | 枚举STemplateType | 0-8 | 同上 | S1 <br> B |
| - | DTemplateType | D轴shape | 枚举DTemplateType | 0-8 | 0：D_TEMPLATE_UNKNOW <br> 5：ALIGNED_80 <br> 6：ALIGNED_96 <br> 8：ALIGNED_128 | S1 <br> B |

<br>

---

<br>

# 2. FA模板说明

模板根据多核切分轴不同，分为几个模板，优先级如下：

- [2.1 TND模板](#second-first-heading)
- [2.2 TNDSameAB模板](#second-second-heading)
- [2.3 SameAB模板](#second-third-heading)
- [2.4 S1S2模板](#second-fourth-heading)
- [2.5 S1模板](#second-fifth-heading)
- [2.6 B模板](#second-sixth-heading)

按顺序命中一个模板的话，就会走入该模板逻辑，如果未命中，会继续向下搜索其余合适的模板。
<br>

**路由规则**（仅大致参考）

* 不涉及确定性计算场景
* TND场景：
  走TND或TNDSameAB模板
* 普通场景：
  * 非FP32，S2 >= 512 and D % 16 != 0 or D = 96，or S2 > 1024 and 128 < D < 196，走SameAB模板
  * 不满足上述条件的，S2 > 1024，走S1S2模板
  * 其余走S1或B模板（shape很小的情况才会走B模板）
  * 特例情况：SameAB模板不支持FP32

<br>

**模板匹配**

通过输入shape及属性，执行以下脚本可以快速判断用例走入的模板。

```python
from enum import Enum
from params import *

# ======================================================================================================================
# 0. FA前向模板类型枚举
class FATemplate(Enum):
    NONE = "[ERROR] No matching template"
    TND = "[INFO] Enter TND template"
    TNDSameAB = "[INFO] Enter TNDSameAB template"
    SameAB = "[INFO] Enter SameAB template"
    S1S2 = "[INFO] Enter S1S2 template"
    S1 = "[INFO] Enter S1 template"
    B = "[INFO] Enter B template"
# ======================================================================================================================

# ======================================================================================================================
# 1. 入参属性枚举
class LayoutEnum(Enum):
    BSH = 0
    SBH = 1
    BNSD = 2
    TND = 3

class DtypeEnum(Enum):
    FLOAT32 = 1
    BFLOAT16 = 2
    FLOAT16_PRECISION = 3

class SparseEnum(Enum):
    ALL = 0
    NONE = 1
    ANY = 2
    CAUSAL = 3
    BAND = 4
    PREFIX = 5
    BAND_COMPRESS = 6
    RIGHT_DOWN_CAUSAL = 7
    RIGHT_DOWN_CAUSAL_BAND = 8
    BAND_LEFT_UP_CAUSAL = 9
# ======================================================================================================================

# ======================================================================================================================
# 2. 模板选择
templateName = FATemplate.NONE

def AnalyzeShape(qShape, kvShape, layout, headNum, headDim):
    if layout == LayoutEnum.BSH:
        B, S1, H1 = qShape
        B, S2, H2 = kvShape
        N1 = headNum
        N2 = H2 // headDim
        G = H1 // H2
        D = headDim
    elif layout == LayoutEnum.SBH:
        S1, B, H1 = qShape
        S2, B, H2 = kvShape
        N1 = headNum
        N2 = H2 // headDim
        G = H1 // H2
        D = headDim
    else:
        B, N1, S1, D = qShape
        B, N2, S2, D = kvShape
        G = N1 // N2

    return B, N1, N2, G, S1, S2, D

def IsTND(layout, B, S1, S2):
    global templateName
    if layout != LayoutEnum.TND:
        return False
    accumS1 = sum(S1)
    accumS2 = sum(S2)
    maxS1 = max(S1)
    maxS2 = max(S2)
    if (B >= 4 and accumS1 >= 8192 and accumS2 >= 8192 and maxS1 >= 512 and maxS2 >= 512) or \
       (maxS2 >= 5120 and maxS1 >= 5120):
        templateName = FATemplate.TNDSameAB
        return True
    templateName = FATemplate.TND
    return True

def IsSameAB(layout, dtype, S1, S2, D):
    global templateName
    if dtype == DtypeEnum.FLOAT32:
        return False
    if S2 >= 512 and (D % 16 != 0 or D == 96):
        templateName = FATemplate.SameAB
        return True
    if S2 > 1024 and D > 128 and D < 196:
        templateName = FATemplate.SameAB
        return True
    if D == 64 and S2 % 64 != 0 and (S2 > 2048 and S2 < 18432):
        templateName = FATemplate.SameAB
        return True
    return False

def IsS1S2(S2):
    global templateName
    if S2 > 1024:
        templateName = FATemplate.S1S2
        return True
    return False

def IsS1(dtype, N2, G, S1, S2, D):
    global templateName
    dtypeSize = 4 if dtype == DtypeEnum.FLOAT32 else 2
    if S2 > 1024:
        return False
    if S2 > 128:
        templateName = FATemplate.S1
        return True
    alignedS1 = (S1 + 16 - 1) // 16 * 16
    alignedS2 = (S2 + 16 - 1) // 16 * 16
    alignedD = (D + 16 - 1) // 16 * 16
    if (N2 * G * ((alignedS1 + alignedS2) * alignedD + alignedS2) * dtypeSize) >= 256 * 1024 or \
       (N2 * G * (alignedS1 + alignedD) * alignedS2 * dtypeSize) >= 256 * 1024:
        templateName = FATemplate.S1
        return True
    if N2 * G * alignedS1 * alignedS2 * dtypeSize <= 65536 * 2:
        return False
    templateName = FATemplate.S1
    return True

def IsB(dtype, N2, G, S1, S2):
    global templateName
    dtypeSize = 4 if dtype == DtypeEnum.FLOAT32 else 2
    alignedS1 = (S1 + 16 - 1) // 16 * 16
    alignedS2 = (S2 + 16 - 1) // 16 * 16
    if alignedS2 > 128:
        return False
    if N2 * G * alignedS1 * alignedS2 * dtypeSize > 65536 * 2:
        return False
    templateName = FATemplate.B
    return True

def IsAvailableTemplate(layout, dtype, B, N1, N2, G, S1, S2, D):
    return IsTND(layout, B, S1, S2) or IsSameAB(layout, dtype, S1, S2, D) or IsS1S2(S2) or \
           IsS1(dtype, N2, G, S1, S2, D) or IsB(dtype, N2, G, S1, S2)

def SelectFATilingTemplate(qShape, kvShape,
                           layout=LayoutEnum.BNSD, dtype=DtypeEnum.BFLOAT16,
                           headNum=0, headDim=0):
    """
    函数SelectFATilingTemplate，判断FlashAttentionScore模板选择
    layout: 默认值BNSD
    dtype: 默认值BFLOAT16
    shape及参数传入规则见下:
    - layout为BSH，默认传入shape为[B, S1, H1]和[B, S2, H2]，必须传入headNum=N1，headDim=D
    - layout为SBH，默认传入shape为[S1, B, H1]和[S2, B, H2]，必须传入headNum=N1，headDim=D
    - layout为BNSD，默认传入shape为[B, N1, S1, D]和[B, N2, S2, D]，不需要传入headNum和headDim
    - layout为TND，默认传入shape为[B, N1, T1, D]和[B, N2, T2, D]，不需要传入headNum和headDim
    """
    print("[INFO] Start matching template for FlashAttentionScore...\n")
    print("[INFO] Layout", layout)
    print("[INFO] Dtype", dtype)
    print("[INFO] qShape", qShape)
    print("[INFO] kvShape", kvShape)
    B, N1, N2, G, S1, S2, D = AnalyzeShape(qShape, kvShape, layout, headNum, headDim)
    print("[INFO] Analyze shape\n B: {}\n N1: {}, N2: {}, G: {}\n S1: {}, S2: {}\n D: {}\n".format(B, N1, N2, G, S1, S2, D))
    if IsAvailableTemplate(layout, dtype, B, N1, N2, G, S1, S2, D):
        print("[INFO] Template match success!!!")
    print(templateName.value)
# ======================================================================================================================


def main():
    # 自定义输入及属性
    layout = LayoutEnum.TND # [BSH, SBH, BNSD, TND]
    dtype = DtypeEnum.BFLOAT16 # [FLOAT32, BFLOAT16, FLOAT16_PRECISION]

    # 例：layout为BSH
    # qShape = [2, 4096, 1024]
    # kvShape = [2, 4096, 1024]
    # headNum = 16
    # headDim = 64

    # 例：layout为SBH
    # qShape = [4096, 2, 1024]
    # kvShape = [4096, 2, 1024]
    # headNum = 16
    # headDim = 64

    # 例：layout为BNSD
    # qShape = [2, 16, 4096, 96]
    # kvShape = [2, 16, 4096, 96]
    # headNum = 0
    # headDim = 0

    # 例：layout为TND
    qShape = [2, 16, [160, 1273], 64]
    kvShape = [2, 16, [160, 1273], 64]
    headNum = 0
    headDim = 0

    # 模板选择
    SelectFATilingTemplate(qShape, kvShape, layout, dtype, headNum, headDim)


if __name__ == '__main__':
    main()
```
<br>

---

<br>
<a id="second-first-heading"></a>

## 2.1 TND模板

### 2.1.1 TND模板走入规则

```c++
if (layout != TND) {
    return false;
}
if (B >= 4 && accumS1 >= 8192 && accumS2 >= 8192 && maxS1 >= 512 && maxS2 >= 512) ||
   (maxS2 >= 5120 && maxS1 >= 5120) {
    isSameAB = true;
}
return true;
```

### 2.1.2 TND模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S1,
              AxisEnum::S2,
              AxisEnum::NONE,
              ImplMode,
              DataType,
              Layout,
              SparseEnum::ANY,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse)
```

模板base值100000000HGF22CBA943。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | ImplMode | A |  |
| 4 | DataType | B |  |
| 5 | Layout | C |  |
| 8 | HasDropOut | F |  |
| 9 | HasAttenMask | G |  |
| 10 | HasPse | H |  |

<br>

---

<br>
<a id="second-second-heading"></a>

## 2.2 TNDSameAB模板

### 2.2.1 TNDSameAB模板走入规则

```c++
if (layout == TND && isSameAB = true) {
    return true;
}
```

### 2.2.2 TNDSameAB模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S1,
              AxisEnum::NONE,
              AxisEnum::NONE,
              DataType,
              Layout,
              SparseEnum::ANY,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse)
```

模板base值1000000000GFE22BA993。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 7 | HasDropOut | E |  |
| 8 | HasAttenMask | F |  |
| 9 | HasPse | G |  |

<br>

---

<br>

<a id="second-third-heading"></a>

## 2.3 SameAB模板

### 2.3.1 SameAB模板走入规则

```c++
if (dtype == FLOAT32) {
    return false;
}
if (S2 >= 512 && D % 16 != 0) {
    return true;
}
if (S2 > 1024 && D > 128 && D < 196) {
    return true;
}
if (D == 64 && S2 % 64 != 0 && (S2 > 2048 && S2 < 18432)) {
    return true;
}
return false;
```

### 2.3.2 SameAB模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S1,
              AxisEnum::NONE,
              AxisEnum::NONE,
              ImplMode,
              DataType,
              Layout,
              Bmm1Format,
              SparseEnum::ANY,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse,
              MatmulPolicyType)
```

模板base值1000000JIHG22DCBA993。

| TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | ImplMode | A |  |
| 4 | DataType | B |  |
| 5 | Layout | C |  |
| 6 | Bmm1Format | D |  |
| 9 | HasDropOut | G |  |
| 10 | HasAttenMask | H |  |
| 11 | HasPse | I |  |
| 12 | MatmulPolicyType | J |  |

<br>

---

<br>

<a id="second-fourth-heading"></a>

## 2.4 S1S2模板

### 2.4.1 S1S2模板走入规则

```c++
if (S2 > 1024) {
    return true;
}
return false;
```

### 2.4.2 S1S2模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S1,
              AxisEnum::S2,
              AxisEnum::NONE,
              ImplMode,
              DataType,
              Layout,
              Bmm1Format,
              SparseEnum::ANY,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse,
              EnableL1Reuse)
```

模板base值1000000JIHG22DCBA943。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | ImplMode | A |  |
| 4 | DataType | B |  |
| 5 | Layout | C |  |
| 6 | Bmm1Format | D |  |
| 9 | HasDropOut | G |  |
| 10 | HasAttenMask | H |  |
| 11 | HasPse | I |  |
| 12 | EnableL1Reuse | J |  |

<br>

---

<br>

<a id="second-fifth-heading"></a>

## 2.5 S1模板

### 2.5.1 S1模板走入规则

```c++
if (S2 > 1024) {
    return false;
}
if (S2 > 128) {
    return true;
}
if ((N2 * G * ((alignedS1 + alignedS2) * alignedD + alignedS2) * dtypeSize) >= 256K ||
    (N2 * G * (alignedS1 + alignedD) * alignedS2 * dtypeSize) >= 256K) {
    return true;
}
if (N2 * G * alignedS1 * alignedS2 * dtypeSize <= 65536 * 2) {
    return false;
}
return true;
```

### 2.5.2 S1模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S1,
              AxisEnum::D,
              AxisEnum::NONE,
              ImplMode,
              DataType,
              Layout,
              Bmm1Format,
              Bmm2Source,
              SparseEnum::ANY,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse,
              EnableL1Reuse,
              S1TemplateType,
              S2TemplateType,
              DTemplateType)
```

模板base值100NMLKJIH22EDCBA953。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | ImplMode | A |  |
| 4 | DataType | B |  |
| 5 | Layout | C |  |
| 6 | Bmm1Format | D |  |
| 7 | Bmm2Source | E |  |
| 10 | HasDropOut | H |  |
| 11 | HasAttenMask | I |  |
| 12 | HasPse | J |  |
| 13 | EnableL1Reuse | K |  |
| 14 | S1TemplateType | L |  |
| 15 | S2TemplateType | M |  |
| 16 | DTemplateType | N |  |

<br>

---

<br>

<a id="second-sixth-heading"></a>

## 2.6 B模板

### 2.6.1 B模板走入规则

```c++
if (alignedS2 > 128) {
    return false;
}
if (N2 * G * alignedS1 * alignedS2 * dtypeSize > 65536 * 2) {
    return false;
}
return true;
```

### 2.6.2 B模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::NONE,
              AxisEnum::NONE,
              AxisEnum::B,
              ImplMode,
              DataType,
              Layout,
              SparseEnum::NONE,
              PerformanceOrientedEnum::BIG_DOUBLE_BUFFER,
              HasDropOut,
              HasAttenMask,
              HasPse,
              S1TemplateType,
              S2TemplateType,
              DTemplateType)
```

模板base值100000KJIHGF21CBA099。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | ImplMode | A |  |
| 4 | DataType | B |  |
| 5 | Layout | C |  |
| 8 | HasDropOut | F |  |
| 9 | HasAttenMask | G |  |
| 10 | HasPse | H |  |
| 11 | S1TemplateType | I |  |
| 12 | S2TemplateType | J |  |
| 13 | DTemplateType | K |  |

<br>

---

<br>

# 3. FAG TilingKey占位参数说明

FlashAttentionScoreGrad的TilingKey的生成规则为，基于base值（1e19）组装的十进制数字，通过接口`GET_TILINGKEY()`获取。不同模板的占位个数不同，因此base值不同，TilingKey位号也不相同，但基于公共的分类方式。

## 3.1 FAG TilingKey关键参数

最多支持19个参数，当前实现包含以下关键参数，从低位到高位，从右向左依次是：\
UB0、UB1、Block、DataType、Layout、Sparse、MatmulConfig、Mm12IsNZOut、Mm345IsNZOut、HasDropOut、HasPse、HasAttenMask、EnableL1Reuse，各参数说明如下：
<br>

| TilingKey位号 | TilingKey位名称 | 作用 | 表示方式 | 取值范围 | 取值含义 | 涉及模板 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | UB0 | 表示UB核内切分的轴 | 枚举AxisEnum | 0-9 | 0：B <br> 1：N2 <br> 2：G <br> 3：S1 <br> 4：S2 <br> 5：D <br> 9：NONE | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 1 | UB1 | 同上，允许最多切分两根轴 | 枚举AxisEnum | 0-9 | 同上 | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 2 | Block | 核间切分轴 | 枚举AxisEnum | 0-9 | 同上 | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 3 | DataType | 数据类型 | 枚举DtypeEnum | 0-3 | 0：FLOAT16 <br> 1：FLOAT32 <br> 2：BFLOAT16 <br> 3：FLOAT16_PRECISION | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 4 | Layout | 数据格式 | 枚举LayoutEnum | 0-3 | 0：BSH <br> 1：SBH <br> 2：BNSD <br> 3：TND | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 5 | Sparse | sparse mode | 枚举SparseEnum | 0-9 | 0：ALL <br> 1：NONE <br> 2：ANY <br> 3：CAUSAL <br> 4：BAND <br> 5：PREFIX <br> 6：BAND_COMPRESS <br> 7：RIGHT_DOWN_CAUSAL <br> 8：RIGHT_DOWN_CAUSAL_BAND <br> 9：BAND_LEFT_UP_CAUSAL | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 6 | MatmulConfig | MatmulConfig | 枚举MatmulConfig | 0-2 | 0：NULL_CONFIG <br> 1：NORMAL_CONFIG <br> 2：MDL_CONFIG | 4.1 <br> 3.1 <br> 1.2 |
| 7 | Mm12IsNZOut | Mm12输出格式 | 枚举OptionEnum | 0-1 | 同上 | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 8 | Mm345IsNZOut | Mm345输出格式 | 枚举OptionEnum | 0-1 | 同上 | 1.1 <br> 4.1 <br> 3.1 <br> 1.2 <br> SameAB |
| 9 | HasDropOut | 是否支持可选输入drop_mask | bool | 0-1 | 0：False <br> 1：True | 1.1 <br> 1.2 <br> SameAB |
| 10 | HasPse | 是否支持可选输入real_shift | bool | 0-1 | 0：False <br> 1：True | 1.1 <br> 1.2 <br> SameAB |
| 11 | HasAttenMask | 是否支持可选输入atten_mask | bool | 0-1 | 0：False <br> 1：True | 1.1 <br> 1.2 <br> SameAB |
| 12 | EnableL1Reuse | 是否使能L1复用 | 枚举OptionEnum | 0-1 | 0：DISABLE <br> 1：ENABLE | 1.2 |

<br>

## 3.2 FAG TilingKey补充参数

其余场景，可以依次在后面定义自己的位域和值。模板中还包含以下其他可以配置的参数，供参考。
<br>

| TilingKey位号 | TilingKey位名称 | 作用 | 表示方式 | 取值范围 | 取值含义 | 涉及模板 |
| --- | --- | --- | --- | --- | --- | --- |
|  | TNDS1Pingpong | S1是否Pingpong | 枚举OptionEnum | 0-1 | 0：DISABLE <br> 1：ENABLE | 1.1 |
|  | S1TemplateType | S1轴shape | 枚举STemplateType | 0-8 | 0：S_TEMPLATE_UNKNOW <br> 1：ALIGNED_16 <br> 2：ALIGNED_32 <br> 3：ALIGNED_48 <br> 4：ALIGNED_64 <br> 5：ALIGNED_80 <br> 6：ALIGNED_96 <br> 7：ALIGNED_112 <br> 8：ALIGNED_128 | 4.1 <br> 3.1 <br> SameAB |
|  | S2TemplateType | S2轴shape | 枚举STemplateType | 0-8 | 同上 | 4.1 <br> 3.1 <br> SameAB |
|  | DTemplateType | D轴shape | 枚举DTemplateType | 0-8 | 0：D_TEMPLATE_UNKNOW <br> 5：ALIGNED_80 <br> 6：ALIGNED_96 <br> 8：ALIGNED_128 | 4.1 <br> 3.1 <br> SameAB |
|  | IsDeterministic | 是否确定性计算 | int32 | 0-1 | 0：NON_DETERMINISTIC <br> 1：DETERMINISTIC | SameAB |

<br>

---

<br>

# 4. FAG模板说明

模板根据多核切分轴不同，分为几个模板，优先级如下：

* [4.1 4.1模板](#fourth-first-heading) flash_attention_score_grad_tiling_bngs1s2_b.cpp
* [4.2 3.1模板](#fourth-second-heading) flash_attention_score_grad_tiling_ngs1s2_bn.cpp
* [4.3 1.2模板](#fourth-third-heading) flash_attention_score_grad_tiling_s1s2_bn2.cpp
* [4.4 SameAB模板](#fourth-fourth-heading) flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp
* [4.5 1.1模板](#fourth-fifth-heading) flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp

按顺序命中一个模板的话，就会走入该模板逻辑，如果未命中，会继续向下搜索其余合适的模板。

<br>

**路由规则**（仅大致参考）

* 确定性计算场景：
  非FP32或者S1 >= 1024 or S2 >= 1024，走SameAB模板；其余走1.2模板
* TND场景：
  平均S1 >= 1024 and 平均S2 >= 1024，走SameAB模板；其余走1.1模板
* 普通场景：
  * N * G * S1 * S2 < 64 * 128，走4.1模板
  * S1 * S2 < 64 * 128，走3.1模板
  * S1 >= 1024 or S2 >= 1024，走SameAB模板
  * 其余走1.2模板
  * 特例情况：3.1/4.1模板不支持FP32，不支持pse内部生成；SameAB模板不支持FP32

<br>

**模板匹配**

通过输入shape及属性，执行以下脚本可以快速判断用例走入的模板。

```python
from enum import Enum
from params import *

# ======================================================================================================================
# 0. FA反向模板类型枚举
class FAGTemplate(Enum):
    NONE = "[ERROR] No matching template"
    OneAndOne = "[INFO] Enter 1.1 template"
    FourAndOne = "[INFO] Enter 4.1 template"
    ThreeAndOne = "[INFO] Enter 3.1 template"
    OneAndTwo = "[INFO] Enter 1.2 template"
    SameAB = "[INFO] Enter SameAB template"
# ======================================================================================================================

# ======================================================================================================================
# 1. 入参属性枚举
class LayoutEnum(Enum):
    BSH = 0
    SBH = 1
    BNSD = 2
    TND = 3

class DtypeEnum(Enum):
    FLOAT32 = 1
    BFLOAT16 = 2
    FLOAT16_PRECISION = 3

class SparseEnum(Enum):
    ALL = 0
    NONE = 1
    ANY = 2
    CAUSAL = 3
    BAND = 4
    PREFIX = 5
    BAND_COMPRESS = 6
    RIGHT_DOWN_CAUSAL = 7
    RIGHT_DOWN_CAUSAL_BAND = 8
    BAND_LEFT_UP_CAUSAL = 9
# ======================================================================================================================

# ======================================================================================================================
# 2. 模板选择
templateName = FAGTemplate.NONE

def AnalyzeShape(qShape, kvShape, layout, headNum, headDim):
    if layout == LayoutEnum.BSH:
        B, S1, H1 = qShape
        B, S2, H2 = kvShape
        N1 = headNum
        N2 = H2 // headDim
        G = H1 // H2
        D = headDim
    elif layout == LayoutEnum.SBH:
        S1, B, H1 = qShape
        S2, B, H2 = kvShape
        N1 = headNum
        N2 = H2 // headDim
        G = H1 // H2
        D = headDim
    else:
        B, N1, S1, D = qShape
        B, N2, S2, D = kvShape
        G = N1 // N2

    return B, N1, N2, G, S1, S2, D

def IsDeterministic(isDeterministic, dtype, S1, S2):
    global templateName
    if not isDeterministic:
        return False
    if dtype == DtypeEnum.FLOAT32:
        templateName = FAGTemplate.OneAndTwo
        return True
    if S1 >= 1024 or S2 >= 1024:
        return False
    templateName = FAGTemplate.OneAndTwo
    return True

def IsDeterministicSameAB(isDeterministic, dtype):
    global templateName
    if not isDeterministic:
        return False
    if dtype == DtypeEnum.FLOAT32:
        return False
    templateName = FAGTemplate.SameAB
    return True

def IsTND(layout, B, S1, S2, sparseMode, isPse):
    global templateName
    if layout != LayoutEnum.TND:
        return False
    accumS1 = sum(S1)
    accumS2 = sum(S2)
    maxS1 = max(S1)
    maxS2 = max(S2)
    lenTND = len(S1)
    if B * maxS1 == accumS1 and B * maxS2 == accumS2 and maxS1 == maxS2 and \
        not(sparseMode == SparseEnum.RIGHT_DOWN_CAUSAL_BAND or sparseMode == SparseEnum.BAND_LEFT_UP_CAUSAL) and \
        not isPse:
        templateName = FAGTemplate.OneAndTwo
        return True
    if accumS1 // lenTND >= 1024 and accumS2 // lenTND >= 1024:
        templateName = FAGTemplate.SameAB
    else:
        templateName = FAGTemplate.OneAndOne
    return True

def IsFourAndOne(dtype, N1, G, S1, S2, isPse, pseType):
    global templateName
    if dtype == DtypeEnum.FLOAT32:
        return False
    if isPse and pseType != 1:
        return False
    bestBasicBlockNum = 64 * 128 // 4 * 3 if S1 >= 4 else 64 * 128
    alignedS1 = (S1 + 16 - 1) // 16 * 16
    alignedS2 = (S2 + 16 - 1) // 16 * 16
    if N1 * G * alignedS1 * alignedS2 <= bestBasicBlockNum:
        templateName = FAGTemplate.FourAndOne
        return True
    return False

def IsThreeAndOne(dtype, G, S1, S2, isPse, pseType):
    global templateName
    if dtype == DtypeEnum.FLOAT32:
        return False
    if isPse and pseType != 1:
        return False
    alignedS2DtypeSize = (S2 * 4 + 16 - 1) // 16 * 16
    if G * S1 * alignedS2DtypeSize == 0 or G != 1 or alignedS2DtypeSize > 1536:
        return False
    if G * S1 * alignedS2DtypeSize <= 32768:
        templateName = FAGTemplate.ThreeAndOne
        return True
    return False

def IsOneAndTwo(B, N2, G, S1, S2, aivBlockDim):
    global templateName
    if S1 >= 1024 or S2 >= 1024:
        return False
    if G > 1 and B * N2 * 2 <= aivBlockDim:
        return False
    if B * N2 < aivBlockDim and S1 > 768 or S2 > 768:
        return False
    templateName = FAGTemplate.OneAndTwo
    return True

def IsSameAB(dtype):
    global templateName
    if dtype == DtypeEnum.FLOAT32:
        return False
    templateName = FAGTemplate.SameAB
    return True

def IsOneAndOne():
    global templateName
    templateName = FAGTemplate.OneAndOne
    return True

def IsAvailableTemplate(isDeterministic, layout, dtype, B, N1, N2, G, S1, S2, aivBlockDim, sparseMode, isPse, pseType):
    return IsDeterministic(isDeterministic, dtype, S1, S2) or IsDeterministicSameAB(isDeterministic, dtype) or \
           IsTND(layout, B, S1, S2, sparseMode, isPse) or IsFourAndOne(dtype, N1, G, S1, S2, isPse, pseType) or \
           IsThreeAndOne(dtype, G, S1, S2, isPse, pseType) or IsOneAndTwo(B, N2, G, S1, S2, aivBlockDim) or \
           IsSameAB(dtype) or IsOneAndOne()

def SelectFAGTilingTemplate(qShape, kvShape,
                            layout=LayoutEnum.BNSD, dtype=DtypeEnum.BFLOAT16,
                            headNum=0, headDim=0, isDeterministic=False,
                            aivBlockDim=40, sparseMode=0,
                            isPse=False, pseType=1):
     """
    函数SelectFAGTilingTemplate，判断FlashAttentionScoreGrad模板选择
    isDeterministic: 默认值False
    layout: 默认值BNSD
    dtype: 默认值BFLOAT16
    aivBlockDim: 默认值40，根据芯片Vector核数配置
    sparseMode: 默认值0
    isPse: 默认值False
    pseType: 默认值1
    shape及参数传入规则见下:
    - layout为BSH，默认传入shape为[B, S1, H1]和[B, S2, H2]，必须传入headNum=N1，headDim=D
    - layout为SBH，默认传入shape为[S1, B, H1]和[S2, B, H2]，必须传入headNum=N1，headDim=D
    - layout为BNSD，默认传入shape为[B, N1, S1, D]和[B, N2, S2, D]，不需要传入headNum和headDim
    - layout为TND，默认传入shape为[B, N1, T1, D]和[B, N2, T2, D]，不需要传入headNum和headDim
    """
    print("[INFO] Start matching template for FlashAttentionScoreGrad...\n")
    print("[INFO] Layout", layout)
    print("[INFO] Dtype", dtype)
    print("[INFO] qShape", qShape)
    print("[INFO] kvShape", kvShape)
    B, N1, N2, G, S1, S2, D = AnalyzeShape(qShape, kvShape, layout, headNum, headDim)
    print("[INFO] Analyze shape\n B: {}\n N1: {}, N2: {}, G: {}\n S1: {}, S2: {}\n D: {}\n".format(B, N1, N2, G, S1, S2, D))
    if IsAvailableTemplate(isDeterministic, layout, dtype, B, N1, N2, G, S1, S2, aivBlockDim, sparseMode, isPse, pseType):
        print("[INFO] Template match success!!!")
    print(templateName.value)
# ======================================================================================================================


def main():
    # 自定义输入及属性
    layout = LayoutEnum.TND # [BSH, SBH, BNSD, TND]
    dtype = DtypeEnum.BFLOAT16 # [FLOAT32, BFLOAT16, FLOAT16_PRECISION]

    # 例：layout为BSH
    # qShape = [2, 4096, 1024]
    # kvShape = [2, 4096, 1024]
    # headNum = 16
    # headDim = 64

    # 例：layout为SBH
    # qShape = [4096, 2, 1024]
    # kvShape = [4096, 2, 1024]
    # headNum = 16
    # headDim = 64

    # 例：layout为BNSD
    # qShape = [2, 16, 4096, 96]
    # kvShape = [2, 16, 4096, 96]
    # headNum = 0
    # headDim = 0

    # 例：layout为TND
    qShape = [2, 16, [160, 1273], 64]
    kvShape = [2, 16, [160, 1273], 64]
    headNum = 0
    headDim = 0

    isDeterministic = False
    aivBlockDim = 40
    sparseMode = 0
    isPse = False
    pseType = 1

    # 模板选择
    SelectFAGTilingTemplate(qShape, kvShape, layout, dtype, headNum, headDim,
                            isDeterministic, aivBlockDim, sparseMode, isPse, pseType)


if __name__ == '__main__':
    main()
```

<br>

---

<br>

<a id="fourth-first-heading"></a>

## 4.1 4.1模板

### 4.1.1 4.1模板走入规则

```c++
if (dtype == FLOAT32) {
    return false;
}
if (isPse && pseType != 1) {
    return false;
}
bestBasicBlockNum = S1 >= 4 ? 64 * 128 / 4 * 3 : 64 * 128;
if (N1 * G * alignedS1 * alignedS2) <= bestBasicBlockNum) {
    return true;
}
return false;
```

### 4.1.2 4.1模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::NONE,
              AxisEnum::NONE,
              AxisEnum::B,
              DataType,
              Layout,
              SparseEnum::NONE,
              MatmulConfig,
              Mm12IsNZOut,
              Mm345IsNZOut,
              S1TemplateType,
              S2TemplateType,
              DTemplateType)
```

模板base值10000000IHGFED1BA099。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 6 | MatmulConfig | D |  |
| 7 | MmPreIsNZOut | E |  |
| 8 | MmNextIsNZOut | F |  |
| 9 | S1TemplateType | G |  |
| 10 | S2TemplateType | H |  |
| 11 | DTemplateType | I |  |

<br>

---

<br>

<a id="fourth-second-heading"></a>

## 4.2 3.1模板

### 4.2.1 3.1模板走入规则

```c++
if (dtype == FLOAT32) {
    return false;
}
if (isPse && pseType != 1) {
    return false;
}
if (G * S1 * alignedS2DtypeSize == 0 || G != 1 || alignedS2DtypeSize > 1536) {
    return false;
}
if (G * S1 * alignedS2DtypeSize <= 32768) {
    return true;
}
return false;
```

### 4.2.2 3.1模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::NONE,
              AxisEnum::NONE,
              AxisEnum::N2,
              DataType,
              Layout,
              SparseEnum::ALL,
              MatmulConfig,
              Mm12IsNZOut,
              Mm345IsNZOut,
              S1TemplateType,
              S2TemplateType,
              DTemplateType)
```

模板base值10000000IHGFED0BA199。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 6 | MatmulConfig | D |  |
| 7 | MmPreIsNZOut | E |  |
| 8 | MmNextIsNZOut | F |  |
| 9 | S1TemplateType | G |  |
| 10 | S2TemplateType | H |  |
| 11 | DTemplateType | I |  |

<br>

---

<br>

<a id="fourth-third-heading"></a>

## 4.3 1.2模板

### 4.3.1 1.2模板走入规则

```c++
if (S1 >= 1024 || S2 >= 1024) {
    return false;
}
if (G > 1 && B * N2 * 2 <= vectorBlockDim) {
    return false;
}
if (B * N2 < vectorBlockDim && S1  > 768 || S2 > 768) {
    return false;
}
return true;
```

### 4.3.2 1.2模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S2,
              AxisEnum::S1,
              AxisEnum::N2,
              DataType,
              Layout,
              SparseEnum::ALL,
              MatmulConfig,
              Mm12IsNZOut,
              HasPse,
              HasAttenMask,
              HasDropOut,
              Mm345IsNZOut,
              EnableL1Reuse)
```

模板base值1000000JIHGFED0BA134。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 6 | MatmulConfig | D |  |
| 7 | Mm12IsNZOut | E |  |
| 8 | HasPse | F |  |
| 9 | HasAttenMask | G |  |
| 10 | HasDropOut | H |  |
| 11 | Mm345IsNZOut | I |  |
| 11 | EnableL1Reuse | J |  |

<br>

---

<br>

<a id="fourth-fourth-heading"></a>

## 4.4 SameAB模板

### 4.4.1 SameAB模板走入规则

```c++
if (dtype == FLOAT32) {
    return false;
}
if (layout == TND && isSameAB) {
    return true;
}
return true;
```

### 4.4.2 SameAB模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S2,
              AxisEnum::S1,
              AxisEnum::S2,
              DataType,
              Layout,
              SparseEnum::ALL,
              HasDropOut,
              HasPse,
              HasAttenMask,
              Mm12IsNZOut,
              Mm345IsNZOut,
              IsDeterministic,
              IsSameAB,
              S1TemplateType,
              S2TemplateType,
              DTemplateType)
```

模板base值1000MLK1IHGFED0BA434。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 6 | HasDropOut | D |  |
| 7 | HasPse | E |  |
| 8 | HasAttenMask | F |  |
| 9 | Mm12IsNZOut | G |  |
| 10 | Mm345IsNZOut | H |  |
| 11 | IsDeterministic | I |  |
| 13 | S1TemplateType | K |  |
| 14 | S2TemplateType | L |  |
| 15 | DTemplateType | M |  |

<br>

---

<br>

<a id="fourth-fifth-heading"></a>

## 4.5 1.1模板

### 4.5.1 1.1模板走入规则

```c++
if (layout == TND) {
    if (accumS1 / lenTND >= 1024 && accumS2 / lenTND >= 1024) {
        return false;
    }
}
return true;
```

### 4.5.2 1.1模板TilingKey规则

```c++
GET_TILINGKEY(AxisEnum::S2,
              AxisEnum::S1,
              AxisEnum::S2,
              DataType,
              Layout,
              SparseEnum::ALL,
              HasDropOut,
              HasPse,
              HasAttenMask,
              Mm12IsNZOut,
              Mm345IsNZOut,
              OptionEnum::DISABLE,
              OptionEnum::DISABLE,
              TNDS1Pingpong)
```

模板base值100000K00HGFED0BA434。

| 模板中TilingKey位号 | TilingKey位名称 | 模板中位号 | 配置规则 |
| --- | --- | --- | --- |
| 3 | DataType | A |  |
| 4 | Layout | B |  |
| 6 | HasDropOut | D |  |
| 7 | HasPse | E |  |
| 8 | HasAttenMask | F |  |
| 9 | Mm1IsNZOut | G |  |
| 10 | Mm2IsNZOut | H |  |
| 13 | TNDS1Pingpong | K |  |
