/**
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
 * \file mm1_processor.h
 * \brief
 */
#ifndef MM2_PROCESSOR_H
#define MM2_PROCESSOR_H

struct Mm2TaskParam {
    uint32_t sigM;
    uint32_t sigN;
    uint32_t sigK;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    int64_t tensorBOffset;
};
template <typename IFAT, typename mmType>
class Mm2Processor
{
public:
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using MM_OUT_T = T;
    using MM_IN_T = Q_T;
    __aicore__ inline Mm2Processor(){};
    // 非量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T>& mm2OutResUb, TSCM<TPosition::VECIN, 1>& softmaxResScmQueue,
                                GlobalTensor<KV_T>& valueGm, const Mm2TaskParam& taskParam);
    // 伪量化
    __aicore__ inline void Send(LocalTensor<MM_OUT_T>& mm2OutResUb, TSCM<TPosition::VECIN, 1> &softmaxResScmQueue,
                                TSCM<TPosition::VECIN, 1>& valueScmQueue, const Mm2TaskParam& taskParam);

    __aicore__ inline void Wait();
    mmType bmm2;
};

template <typename IFAT, typename mmType>
__aicore__ inline void Mm2Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T>& mm2OutResUb,
                                                        TSCM<TPosition::VECIN, 1> &softmaxResScmQueue,
                                                        GlobalTensor<KV_T>& valueGm, const Mm2TaskParam& taskParam)
{
    LocalTensor<Q_T> softmaxResScmTensor = softmaxResScmQueue.DeQue<Q_T>();
    bmm2.SetOrgShape(taskParam.orgM, taskParam.orgN, taskParam.orgKa, taskParam.orgKb, taskParam.orgKc);
    bmm2.SetTensorA(softmaxResScmTensor);
    bmm2.SetTensorB(valueGm[taskParam.tensorBOffset]);
    bmm2.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    bmm2.template IterateAll<false>(mm2OutResUb, false, false, true);

    softmaxResScmQueue.FreeTensor(softmaxResScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm2Processor<IFAT, mmType>::Send(LocalTensor<MM_OUT_T>& mm2OutResUb,
                                                        TSCM<TPosition::VECIN, 1>& softmaxResScmQueue,
                                                        TSCM<TPosition::VECIN, 1>& valueScmQueue,
                                                        const Mm2TaskParam& taskParam)
{
    LocalTensor<Q_T> softmaxResScmTensor = softmaxResScmQueue.DeQue<Q_T>();
    LocalTensor<Q_T> valueScmTensor = valueScmQueue.DeQue<Q_T>();

    bmm2.SetTensorA(softmaxResScmTensor);
    bmm2.SetTensorB(valueScmTensor);
    bmm2.SetTail(taskParam.sigM, taskParam.sigN, taskParam.sigK);
    bmm2.template IterateAll<false>(mm2OutResUb, false, false, true);
    
    valueScmQueue.FreeTensor(valueScmTensor);
    softmaxResScmQueue.FreeTensor(softmaxResScmTensor);
}

template <typename IFAT, typename mmType>
__aicore__ inline void Mm2Processor<IFAT, mmType>::Wait()
{
    bmm2.WaitIterateAll();
    bmm2.End();
}
#endif
