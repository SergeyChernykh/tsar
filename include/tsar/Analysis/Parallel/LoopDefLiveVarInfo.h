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
//===----------------------------------------------------------------------===//
//
// This file defines passes to finds defs variable in loops and
// live variables after loop.
//
//===----------------------------------------------------------------------===//

#ifndef TSAR_ANALYSIS_LOOPDEFLIVEVARINFO_H
#define TSAR_ANALYSIS_LOOPDEFLIVEVARINFO_H

#include "tsar/Analysis/Parallel/Passes.h"
#include "tsar/Analysis/Memory/DIEstimateMemory.h"

#include "bcl/utility.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/Pass.h>

namespace llvm {
  class Loop;
}

namespace tsar {
  /// List of loops which could be executed in a parallel way.
  using LoopDefLiveInfo = llvm::DenseMap<llvm::MDNode*, //loopid
    std::pair<
    llvm::DenseSet<llvm::DIVariable*>,
    llvm::DenseSet<llvm::DIVariable*>>>;
}

namespace llvm {
  /// Determine loops which could be executed in a parallel way.
  class LoopDefLiveVarInfoPass : public FunctionPass, private bcl::Uncopyable {
  public:
    static char ID;

    LoopDefLiveVarInfoPass() : FunctionPass(ID) {
      initializeLoopDefLiveVarInfoPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function& F) override;
    void getAnalysisUsage(AnalysisUsage& AU) const override;

    void releaseMemory() override { mLoopDefLiveInfo.clear(); }

    /// Return list of loops which could be executed in a parallel way.
    tsar::LoopDefLiveInfo& getLoopDefLiveVarInfoInfo() { return mLoopDefLiveInfo; }

    /// Return list of loops which could be executed in a parallel way.
    const tsar::LoopDefLiveInfo& getLoopDefLiveVarInfo() const {
      return mLoopDefLiveInfo;
    }

  private:
    tsar::LoopDefLiveInfo mLoopDefLiveInfo;
  };
}

#endif//TSAR_ANALYSIS_LOOPDEFLIVEVARINFO_H
