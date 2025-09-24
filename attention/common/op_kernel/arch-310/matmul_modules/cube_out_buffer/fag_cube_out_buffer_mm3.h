/*	
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.	
 *	
 * Licensed under the Apache License, Version 2.0 (the "License");	
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at	
 *	
 * http://www.apache.org/licenses/LICENSE-2.0	
 *	
 * Unless required by applicable law or agreed to in writing, software	
 * distributed under the License is distributed on an "AS IS" BASIS,	
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.	
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file fag_cube_out_buffer.h
 * \brief
 */
#ifndef FAG_CUBE_OUT_BUFFER_MM3_H
#define FAG_CUBE_OUT_BUFFER_MM3_H

#include "../fag_flag_data.h"
namespace AscendC {
namespace Impl {
namespace Detail {
template <typename IMPL, typename L0cT, const auto &MM_CFG, typename = void> class FagCubeOutBufferMM3 {
    using L0cT_ = typename L0cT::T;
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

public:
    __aicore__ inline FagCubeOutBufferMM3(){};
    __aicore__ inline void Init(uint32_t lenFactor = 1, int32_t cacheSize = 1)
    {
    }

    __aicore__ inline LocalTensor<L0cT_> AllocTensor()
    {
        cMatrix_ = gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN]
                       .l0cQue.template AllocTensor<L0cT_>();
        gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN].srcAddr = cMatrix_.address_;
        return cMatrix_;
    }

    __aicore__ inline LocalTensor<L0cT_> GetTensor()
    {
        // 返回值为L0C Tensor
        LocalTensor<L0cT_> l0cBuf;
        l0cBuf.SetAddr(gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN].srcAddr);
        return l0cBuf;
    }

    __aicore__ inline void EnQue(LocalTensor<L0cT_> &tensor)
    {
        gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN].l0cQue.EnQue(tensor);
    }

    __aicore__ inline LocalTensor<L0cT_> DeQue()
    {
        auto co1Local = gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN]
                            .l0cQue.template DeQue<L0cT_>();
        return co1Local;
    }

    __aicore__ inline void FreeTensor(LocalTensor<L0cT_> &co1Local)
    {
        gL0cArray[MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData().nextMorN].l0cQue.FreeTensor(co1Local);
    }

    __aicore__ inline void Destroy()
    {
    }

private:
    LocalTensor<L0cT_> cMatrix_;
};

} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif