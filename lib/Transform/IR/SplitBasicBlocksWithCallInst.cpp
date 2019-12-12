//===--- SplitBasicBlocksWithCallInst.cpp --- Split Basic Block Tranform ----------*- C++ -*-===//
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
// This file implements a pass to extract each call instruction (except debug instructions) into its own new basic block.
//
//===----------------------------------------------------------------------===//
#include "tsar/Transform/IR/Passes.h"
#include <bcl/utility.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
using namespace llvm;

namespace {
class SplitBasicBlocksWithCallInstPass : public FunctionPass, private bcl::Uncopyable {
public:
  static char ID;
  SplitBasicBlocksWithCallInstPass() :FunctionPass(ID) {
    initializeSplitBasicBlocksWithCallInstPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& F) override;

  /// Specifies a list of analyzes  that are necessary for this pass.
  void getAnalysisUsage(AnalysisUsage& AU) const {};

};
}


#undef DEBUG_TYPE
#define DEBUG_TYPE "Split-BB"

char SplitBasicBlocksWithCallInstPass::ID = 0;
INITIALIZE_PASS_BEGIN(SplitBasicBlocksWithCallInstPass, "Split-BB",
  "Split BB in call instraction", false, true)
INITIALIZE_PASS_END(SplitBasicBlocksWithCallInstPass, "Split-BB",
    "Split BB in call instraction", false, true)

FunctionPass * llvm::createSplitBasicBlocksWithCallInstPass() {
  return new SplitBasicBlocksWithCallInstPass();
}

bool SplitBasicBlocksWithCallInstPass::runOnFunction(Function& F) {
  LLVM_DEBUG(
    dbgs() << "[SPLIT_BASIC_BLOCK_WITH_CALL_INST]: "
      << "Begin of SplitBasicBlocksWithCallInstPass\n";
    dbgs() << "[SPLIT_BASIC_BLOCK_WITH_CALL_INST]: " << F.getName() << " Befor transform\n";
    F.dump();
  );
  if (F.hasName() && !F.empty()) {
    for (auto currBB = F.begin(), lastBB = F.end(); currBB != lastBB; ++currBB) {
      LLVM_DEBUG(
        dbgs() << "[SPLIT_BASIC_BLOCK_WITH_CALL_INST]: "
          << "Current BBname: " << currBB->getName() << "\n";
      );
      TerminatorInst* ti = currBB->getTerminator();

      if (ti != nullptr) {
        for (auto currInstr = currBB->begin(), lastInstr = currBB->end();
          currInstr != lastInstr; ++currInstr) {
          Instruction* i = &*currInstr;
          if (i == ti)
            break;
          BasicBlock* newBB;
          if (auto* callInst = dyn_cast<CallInst>(i)) {
              if (i == &*(currBB->begin()))
                newBB = &*currBB;
              else
                newBB = currBB->splitBasicBlock(callInst);
            auto nextInstr = callInst->getNextNonDebugInstruction();
            if (nextInstr != ti) {
              newBB->splitBasicBlock(nextInstr);
              currBB++;
              break;
            }
          }
        }
      }
    }
  }
  dbgs() << "[SPLIT_BASIC_BLOCK_WITH_CALL_INST]: " << F.getName() << " After transform\n";
  F.dump();
  return true;
}

