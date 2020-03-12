//===- LoopDefLiveVarInfo.h -- Find Def and Live vars for loops-*- C++ -*-===//
//
//                       Traits Static Analyzer (SAPFOR)
//
// Copyright 2019 DVM System Group
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===---------------------------------------------------------------------===//
//
// This file defines passes to finds defs variable in loops and
// live variables after loop.
//
//===---------------------------------------------------------------------===//

#include "tsar/Analysis/Parallel/LoopDefLiveVarInfo.h"
#include "tsar/Analysis/Parallel/ParallelLoop.h"
#include "tsar/Analysis/DFRegionInfo.h"
#include "tsar/Analysis/Memory/LiveMemory.h"
#include "tsar/Analysis/Memory/EstimateMemory.h"
#include <llvm/IR/Dominators.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "loop-dl-info"
using namespace llvm;
using namespace tsar;

char LoopDefLiveVarInfoPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopDefLiveVarInfoPass, "loop-dl-info",
  "Def Live Loop Analysis", true, true)
  INITIALIZE_PASS_DEPENDENCY(ParallelLoopPass)
  INITIALIZE_PASS_DEPENDENCY(DFRegionInfoPass)
  INITIALIZE_PASS_DEPENDENCY(LiveMemoryPass)
  INITIALIZE_PASS_DEPENDENCY(DefinedMemoryPass)
  INITIALIZE_PASS_DEPENDENCY(EstimateMemoryPass)
  INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(DIEstimateMemoryPass)

INITIALIZE_PASS_END(LoopDefLiveVarInfoPass, "loop-dl-info",
  "Def Live Loop Analysis",true, true)

void LoopDefLiveVarInfoPass::getAnalysisUsage(AnalysisUsage& AU) const {
  AU.addRequired<ParallelLoopPass>();
  AU.addRequired<DFRegionInfoPass>();
  AU.addRequired<LiveMemoryPass>();
  AU.addRequired<DefinedMemoryPass>();
  AU.addRequired<EstimateMemoryPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<DIEstimateMemoryPass>();
  AU.setPreservesAll();
}

void FindVars(DenseSet<DIVariable*>& VarSet, Function& F,
  MemorySet<MemoryLocationRange> MemSet, AliasTree& AT, DominatorTree& DT,
  DataLayout DL, DIAliasTree& DIAT) {
  for (auto& MLR : MemSet) {
    auto* EM = AT.find(MLR);
    assert(EM && "Estimate memory must be presented in alias tree!");
    auto RawDIM = getRawDIMemoryIfExists(
      *EM->getTopLevelParent(), F.getContext(), DL, DT);
    assert(RawDIM && "Unknown raw memory!");
    assert(DIAT.find(*RawDIM) != DIAT.memory_end() &&
      "Memory must exist in alias tree!");
    auto& DIEM = cast<DIEstimateMemory>(*DIAT.find(*RawDIM));
    auto Var = DIEM.getVariable();;
    VarSet.insert(Var);
  }
}

bool LoopDefLiveVarInfoPass::runOnFunction(Function& F) {
  auto& PL = getAnalysis<ParallelLoopPass>().getParallelLoopInfo();
  auto& RI = getAnalysis<DFRegionInfoPass>().getRegionInfo();
  auto& LM = getAnalysis<LiveMemoryPass>().getLiveInfo();
  auto& DM = getAnalysis<DefinedMemoryPass>().getDefInfo();
  auto& AT = getAnalysis<EstimateMemoryPass>().getAliasTree();
  auto& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto& DIAT = getAnalysis<DIEstimateMemoryPass>().getAliasTree();
  auto& DL = F.getParent()->getDataLayout();

  for (auto LIter = PL.begin(), End = PL.end(); LIter != End; LIter++) {
    auto L = *LIter;
    auto LID = L->getLoopID();
    if (!mLoopDefLiveInfo.count(LID)) {
      mLoopDefLiveInfo.try_emplace(LID);
    }
    auto Region = RI.getRegionFor(L);
    auto& DefsR = (*std::get<0>(DM[Region])).getDefs();
    auto& MayDefsR = (*std::get<0>(DM[Region])).getMayDefs();
    auto& UsesR = (*std::get<0>(DM[Region])).getUses();
    auto& LMR = LM[Region].get()->getOut();
    FindVars(mLoopDefLiveInfo[LID].first, F, DefsR, AT, DT, DL, DIAT);
    FindVars(mLoopDefLiveInfo[LID].first, F, MayDefsR, AT, DT, DL, DIAT);
    FindVars(mLoopDefLiveInfo[LID].first, F, UsesR, AT, DT, DL, DIAT);
    FindVars(mLoopDefLiveInfo[LID].second, F, LMR, AT, DT, DL, DIAT);
  }
  return false;
}