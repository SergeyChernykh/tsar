//===-- DVMHSMAutoPar.cpp - OpenMP Based Parallelization (Clang) -*- C++ -*===//
//
//                       Traits Static Analyzer (SAPFOR)
//
// Copyright 2020 DVM System Group
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
// This file implements a pass to perform DVMH-based auto parallelization for
// shared memory.
//
//===----------------------------------------------------------------------===//

#include "SharedMemoryAutoPar.h"
#include "tsar/Analysis/DFRegionInfo.h"
#include "tsar/Analysis/Clang/ASTDependenceAnalysis.h"
#include "tsar/Analysis/Clang/CanonicalLoop.h"
#include "tsar/Analysis/Clang/LoopMatcher.h"
#include "tsar/Analysis/Passes.h"
#include "tsar/Analysis/KnownFunctionTraits.h"
#include "tsar/Analysis/Parallel/Passes.h"
#include "tsar/Analysis/Parallel/ParallelLoop.h"
#include "tsar/Analysis/Memory/MemoryAccessUtils.h"
#include "tsar/Analysis/Memory/DIClientServerInfo.h"
#include "tsar/Core/Query.h"
#include "tsar/Core/TransformationContext.h"
#include "tsar/Support/Clang/Utils.h"
#include "tsar/Transform/Clang/Passes.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <string>
#include <algorithm>

using namespace clang;
using namespace llvm;
using namespace tsar;

#undef DEBUG_TYPE
#define DEBUG_TYPE "clang-dvmh-sm-parallel"

namespace {

using ReductionVarListT = ClangDependenceAnalyzer::ReductionVarListT;
using SortedVarListT = ClangDependenceAnalyzer::SortedVarListT;
using FunctionAnalysisResult = std::tuple <DominatorTree*, PostDominatorTree*,
  TargetLibraryInfo*, AliasTree*, DIMemoryClientServerInfo*,
  const CanonicalLoopSet*, DFRegionInfo*>;

class ParallelRegion;
class ActualRegion;

template<class T>
void unionVarSets(const std::vector<ForStmt const*>& Fors,
  DenseMap<const ForStmt*, ClangDependenceAnalyzer::ASTRegionTraitInfo>& ForVarInfo,
  SortedVarListT& Result) {
  for (auto AST : Fors) {
    for (auto VarName : ForVarInfo[AST].get<T>()) {
      Result.insert(VarName);
    }
  }
}

void unionVarRedSets(const std::vector<ForStmt const*>& Fors,
  DenseMap<const ForStmt*, ClangDependenceAnalyzer::ASTRegionTraitInfo>& ForVarInfo,
  ReductionVarListT& Result) {
  unsigned I = trait::DIReduction::RK_First;
  unsigned EI = trait::DIReduction::RK_NumberOf;
  for (; I < EI; ++I) {
    SortedVarListT tmp;
    for (auto AST : Fors) {
      for (auto VarName : ForVarInfo[AST].get<trait::Reduction>()[I]) {
        tmp.insert(VarName);
      }
    }
    Result[I] = tmp;
  }
}

void addVarList(const SortedVarListT& VarInfoList,
    SmallVectorImpl<char>& Clause) {
  Clause.push_back('(');
  auto I = VarInfoList.begin(), EI = VarInfoList.end();
  Clause.append(I->begin(), I->end());
  for (++I; I != EI; ++I) {
    Clause.append({ ',', ' ' });
    Clause.append(I->begin(), I->end());
  }
  Clause.push_back(')');
}

/// Add clauses for all reduction variables from a specified list to
/// the end of `ParallelFor` pragma.
void addVarList(ReductionVarListT& VarInfoList,
   SmallVectorImpl<char>& ParallelFor) {
  unsigned I = trait::DIReduction::RK_First;
  unsigned EI = trait::DIReduction::RK_NumberOf;
  for (; I < EI; ++I) {
    if (VarInfoList[I].empty())
      continue;
    SmallString<7> RedKind;
    switch (static_cast<trait::DIReduction::ReductionKind>(I)) {
    case trait::DIReduction::RK_Add: RedKind += "sum"; break;
    case trait::DIReduction::RK_Mult: RedKind += "product"; break;
    case trait::DIReduction::RK_Or: RedKind += "or"; break;
    case trait::DIReduction::RK_And: RedKind += "and"; break;
    case trait::DIReduction::RK_Xor: RedKind + "xor "; break;
    case trait::DIReduction::RK_Max: RedKind += "max"; break;
    case trait::DIReduction::RK_Min: RedKind += "min"; break;
    default: llvm_unreachable("Unknown reduction kind!"); break;
    }
    ParallelFor.append({ 'r', 'e', 'd', 'u', 'c', 't', 'i', 'o', 'n' });
    ParallelFor.push_back('(');
    auto VarItr = VarInfoList[I].begin(), VarItrE = VarInfoList[I].end();
    auto k = VarItr->begin();
    ParallelFor.append(RedKind.begin(), RedKind.end());
    ParallelFor.push_back('(');
    ParallelFor.append(VarItr->begin(), VarItr->end());
    ParallelFor.push_back(')');
    for (++VarItr; VarItr != VarItrE; ++VarItr) {
      ParallelFor.push_back(',');
      ParallelFor.append(RedKind.begin(), RedKind.end());
      ParallelFor.push_back('(');
      ParallelFor.append(VarItr->begin(), VarItr->end());
      ParallelFor.push_back(')');
    }
    ParallelFor.push_back(')');
  }
}

/// This class stores information about loops in one region
class ParallelRegion {
public:

  ParallelRegion(const ForStmt* AST, Loop* IR, const Stmt* Parent,
      Function* F,
      ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
    mLoops.push_back(AST);
    mParent = Parent;

    mIrLoops[AST] = IR;
    mVarInfo[AST] = VarInfo;
    mHostInfo[AST] = IsHost;
    mFunction = F;
  }

  SourceLocation GetLocStart() {
    Sort();
    return mLoops[0]->getLocStart();
  }

  SourceLocation GetLocEnd() {
    Sort();
    return mLoops[mLoops.size() - 1]->getLocEnd();
  }

  const ForStmt* GetLoop() const {
    assert(mLoops.size() == 1);
    return mLoops[0];
  }

  std::vector <const ForStmt*>& GetLoops() {
    Sort();
    return mLoops;
  }

  bool GetHostInfo(const ForStmt* AST) {
    assert(mHostInfo.count(AST));
    return mHostInfo[AST];
  }

  bool GetUnionHostInfo() {
    bool IsHost = false;
    for (auto& item : mHostInfo) {
      IsHost |= item.second;
    }
    return IsHost;
  }

  Loop* GetIrLoop(const ForStmt* AST) {
    assert(mIrLoops.count(AST));
    return mIrLoops[AST];
  }

  const ForStmt* GetLastLoop() {
    Sort();
    return mLoops[mLoops.size() - 1];
  }

  const ForStmt* GetFirstLoop() {
    Sort();
    return mLoops[0];
  }

  ClangDependenceAnalyzer::ASTRegionTraitInfo GetVarInfo(const ForStmt* AST) {
    assert(mVarInfo.count(AST));
    return mVarInfo[AST];
  }

  DenseMap<const ForStmt*,
    ClangDependenceAnalyzer::ASTRegionTraitInfo> GetVarInfo() {
    return mVarInfo;
  }

  const Stmt* GetParent() const {
    return mParent;
  }

  void AppendParallelRegion(ParallelRegion& Region) {
    assert(mParent == Region.GetParent());
    auto AST = Region.GetLoop();
    assert(std::find(begin(mLoops), end(mLoops), AST) == mLoops.end());
    assert(!mHostInfo.count(AST) && !mIrLoops.count(AST) &&
      !mVarInfo.count(AST));
    mLoops.push_back(AST);
    mHostInfo[AST] = Region.GetHostInfo(AST);
    mIrLoops[AST] = Region.GetIrLoop(AST);
    mVarInfo[AST] = Region.GetVarInfo(AST);
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    InsertAllPragmaParallel(TfmCtx);
    InsertPragmaRegion(TfmCtx);
  }

  void Dump() {
    for (auto& AST : mLoops) {
      dbgs() << "AST:\n";
      AST->dump();
      dbgs() << "IR:\n";
      mIrLoops[AST]->dump();
    }
  }

private:

  void InsertPragmaRegion(TransformationContext& TfmCtx) {
    SmallString<128> DVMHRegion("#pragma dvm region");
    
    if (GetUnionHostInfo()) {
      DVMHRegion += " targets(HOST)";
    }
    else {
      SortedVarListT UnionVarSet;
      unionVarSets<trait::ReadOccurred>(mLoops, mVarInfo, UnionVarSet);
      if (!UnionVarSet.empty()) {
        DVMHRegion += " in";
        addVarList(UnionVarSet, DVMHRegion);
        UnionVarSet.clear();
      }
      unionVarSets<trait::WriteOccurred>(mLoops, mVarInfo, UnionVarSet);
      if (!UnionVarSet.empty()) {
        DVMHRegion += " out";
        addVarList(UnionVarSet, DVMHRegion);
        UnionVarSet.clear();
      }
      unionVarSets<trait::Private>(mLoops, mVarInfo, UnionVarSet);
      if (!UnionVarSet.empty()) {
        DVMHRegion += " local";
        addVarList(UnionVarSet, DVMHRegion);
      }
    }
    DVMHRegion += "\n{\n";

    auto& Rewriter = TfmCtx.getRewriter();
    Rewriter.InsertTextBefore(GetLocStart(), DVMHRegion);
     auto& ASTCtx = TfmCtx.getContext();
     Token SemiTok;
     auto InsertLoc = (!getRawTokenAfter(GetLocEnd(),
       ASTCtx.getSourceManager(), ASTCtx.getLangOpts(), SemiTok)
       && SemiTok.is(tok::semi))
       ? SemiTok.getLocation() : GetLocEnd();
     Rewriter.InsertTextAfterToken(InsertLoc, "}");

  }

  void InsertAllPragmaParallel(TransformationContext& TfmCtx) {
    for (const auto AST : mLoops) {
      InsertPragmaParallel(AST, TfmCtx);
    }
  }

  void InsertPragmaParallel(const ForStmt* AST, TransformationContext& TfmCtx) {
    SmallString<128> ParallelFor("#pragma dvm parallel (1)");
    auto PrivateVarsList = mVarInfo[AST].get<trait::Private>();
    if (!PrivateVarsList.empty()) {
      ParallelFor += " private";
      addVarList(PrivateVarsList, ParallelFor);
    }
    auto ReductionVarsList = mVarInfo[AST].get<trait::Reduction>();
    addVarList(ReductionVarsList, ParallelFor);
    ParallelFor += '\n';

    auto& Rewriter = TfmCtx.getRewriter();
    Rewriter.InsertTextBefore(AST->getLocStart(), ParallelFor);
    
  }

  void Sort() {
    std::sort(mLoops.begin(), mLoops.end(),
      [](const ForStmt* F1, const ForStmt* F2) {
        return F1->getLocStart() < F2->getLocEnd();
      });
  }

private:
  // Loops in region
  std::vector <const ForStmt*> mLoops;

  DenseMap<const ForStmt*, bool> mHostInfo;

  // IR representation of loop
  DenseMap<const ForStmt*, Loop*> mIrLoops;

  // Var info for pragmas
  DenseMap<const ForStmt*,
    ClangDependenceAnalyzer::ASTRegionTraitInfo> mVarInfo;

  const Stmt* mParent;
  Function* mFunction;
};

/// This class stores information about regions in one #pragma dvm actual/get_actual
/// actual (...)
/// region 1
/// region 2
/// get_actual(...)
class ActualRegion {
public:

  ActualRegion(const ForStmt* AST, Loop* IR, const Stmt* Parent,
    Function* F,
    ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
    mRegions.push_back({ AST, IR, Parent, F, VarInfo, IsHost});
    mParent = Parent;
    mFunction = F;
  }

  ActualRegion(const Stmt* Parent, Function* F) {
    mParent = Parent;
    mFunction = F;
  }

  SourceLocation GetLocStart() {
    Sort();
    return mRegions[0].GetLocStart();
  }

  SourceLocation GetLocEnd() {
    Sort();
    return mRegions[mRegions.size() - 1].GetLocEnd();
  }

  const ForStmt* GetLoop() const {
    assert(mRegions.size() == 1);
    return mRegions[0].GetLoop();
  }

  Loop* GetLastIRLoop() {
    Sort();
    auto LR = mRegions[mRegions.size() - 1];
    return LR.GetIrLoop(LR.GetLastLoop());
  }

  Loop* GetFirstIRLoop() {
    Sort();
    auto LR = mRegions[0];
    return LR.GetIrLoop(LR.GetFirstLoop());
  }

  ParallelRegion GetParallelRegion() const {
    assert(mRegions.size() == 1);
    return mRegions[0];
  }

  std::vector<ParallelRegion>& GetParallelRegions() {
    return mRegions;
  }

  void AppendParallelRegion(const ActualRegion& Region) {
    auto ParReg = Region.GetParallelRegion();
    if (mRegions.empty()) {
      mRegions.push_back(ParReg);
    } else {
      mRegions[0].AppendParallelRegion(ParReg);
    }
  }

  bool Empty() const {
    return mRegions.empty();
  }

  void Clear() {
    mRegions.clear();
  }

  const Stmt* GetParent() const {
    return mParent;
  }

  Function* GetFunction() {
    return mFunction;
  }

  ParallelRegion& GetRegion(size_t i) {
    return mRegions[i];
  }

  std::vector<ParallelRegion>& GetRegions() {
    return mRegions;
  }

  void AppendActualRegion(ActualRegion& Region) {
    assert(mParent == Region.GetParent());
    assert(mFunction == Region.GetFunction());
    for (auto PR : Region.GetRegions()) {
      mRegions.push_back(PR);
    }
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    for (auto& Region : mRegions) {
      Region.InsertPragmas(TfmCtx);
    }
    InsertPragmaActual(TfmCtx);
    InsertPragmaGetActual(TfmCtx);
  }

  void Dump() {
    for (auto& region : mRegions) {
      dbgs() << "~~~~~~~~~~~\n";
      region.Dump();
      dbgs() << "~~~~~~~~~~~\n";

    }
  }


private:
  // TODO: extend this function for several regions in actual region
  void InsertPragmaActual(TransformationContext& TfmCtx) {
    SmallString<128> DVMHActual;
    if (!mRegions[0].GetUnionHostInfo()) {
      SortedVarListT UnionVarSet;
      for (auto PR : mRegions) {
        unionVarSets<trait::ReadOccurred>(PR.GetLoops(),
          PR.GetVarInfo(), UnionVarSet);
      }
      if (!UnionVarSet.empty()) {
        DVMHActual += "#pragma dvm actual";
        addVarList(UnionVarSet, DVMHActual);
        DVMHActual += '\n';
      }
    }
    if (!DVMHActual.empty()) {
      auto& Rewriter = TfmCtx.getRewriter();
      Rewriter.InsertTextBefore(GetLocStart(), DVMHActual);
    }
  }

  // TODO: extend this function for several regions in actual region
  void InsertPragmaGetActual(TransformationContext& TfmCtx) {
    SmallString<128> DVMHGetActual;
    if (!mRegions[0].GetUnionHostInfo()) {
      SortedVarListT UnionVarSet;
      for (auto PR : mRegions) {
        unionVarSets<trait::WriteOccurred>(PR.GetLoops(),
          PR.GetVarInfo(), UnionVarSet);
      }
      if (!UnionVarSet.empty() || true) {
        DVMHGetActual += "#pragma dvm get_actual";
        addVarList(UnionVarSet, DVMHGetActual);
        DVMHGetActual += '\n';
      }

      auto& Rewriter = TfmCtx.getRewriter();
      auto& ASTCtx = TfmCtx.getContext();
      Token SemiTok;
      auto InsertLoc = (!getRawTokenAfter(GetLocEnd(),
        ASTCtx.getSourceManager(), ASTCtx.getLangOpts(), SemiTok)
        && SemiTok.is(tok::semi))
        ? SemiTok.getLocation() : GetLocEnd();
      if (!DVMHGetActual.empty()) {
        Rewriter.InsertTextAfterToken(InsertLoc, "\n");
        Rewriter.InsertTextAfterToken(InsertLoc, DVMHGetActual);
      }
    }
  }

  void Sort() {
    std::sort(mRegions.begin(), mRegions.end(),
      [](ParallelRegion& R1, ParallelRegion& R2) {
        return R1.GetLocStart() < R2.GetLocStart();
      });
  }

private:
  //Regions in actual regions
  std::vector<ParallelRegion> mRegions;

  const Stmt* mParent;
  Function* mFunction;
};

bool findMemoryOverlap(ActualRegion& R1, ActualRegion& R2,
    TargetLibraryInfo* TLI, AliasTree* AT, DIMemoryClientServerInfo* DIMInfo,
    DominatorTree* DT, const CanonicalLoopSet* CL, DFRegionInfo* RI) {
  auto L1 = R1.GetLastIRLoop();
  auto L2 = R2.GetFirstIRLoop();
  assert(R1.GetFunction() == R2.GetFunction());
  auto F = R1.GetFunction();
  bool skip = true;
  bool Overlap = false;
  for (auto& CBB = F->begin(), EBB = F->end(); CBB != EBB; CBB++) {
     if (L1->contains(&*CBB) || L1->getExitBlock() == (&*CBB)) {
      skip = false;
      continue;
    }
    if (skip) {
      continue;
    }
    if (L2->contains(&*CBB) || L2->getHeader() == (&*CBB)) {
      continue;
    }
    auto CanonicalItr = CL->find_as(RI->getRegionFor(L2));
    assert(CanonicalItr != CL->end() && (**CanonicalItr).isCanonical());
    for (auto& I : CBB->instructionsWithoutDebug()) {
      if (auto II = llvm::dyn_cast<IntrinsicInst>(&I))
        if (isMemoryMarkerIntrinsic(II->getIntrinsicID()) ||
          isDbgInfoIntrinsic(II->getIntrinsicID()))
          continue;
      if (isa<StoreInst>(I) && (*CanonicalItr)->getInduction() == I.getOperand(1)) {
        continue;
      }
      for_each_memory(I, *TLI,
         [&Overlap, AT, DIMInfo, DT, &R1, &R2](Instruction& I, MemoryLocation&& Loc,
           unsigned Idx, AccessInfo, AccessInfo W) {
             if (W == AccessInfo::No)
               return;
             auto EM = AT->find(Loc);
             assert(EM && "Estimate memory location must not be null!");
             auto& DL = I.getModule()->getDataLayout();
             DIMemory* DIM = DIMInfo->findFromClient(
               *EM->getTopLevelParent(), DL, *DT).get<Clone>();
             for (auto& PR : R1.GetParallelRegions()) {
               for (auto AST : PR.GetLoops()) {
                 Loop* IR = PR.GetIrLoop(AST);
                 auto DIMIR = DIMInfo->findFromClient(*IR);
                 for (auto AT : *DIMIR) {
                   if (AT.find(DIM) != AT.end()) {
                     Overlap = true;
                     return;
                   }
                 }
               }
             }
             for (auto& PR : R2.GetParallelRegions()) {
               for (auto AST : PR.GetLoops()) {
                 Loop* IR = PR.GetIrLoop(AST);
                 auto DIMIR = DIMInfo->findFromClient(*IR);
                 for (auto AT : *DIMIR) {
                   if (AT.find(DIM) != AT.end()) {
                     Overlap = true;
                     return;
                   }
                 }
               }
             }
         },
         [](Instruction& I, AccessInfo, AccessInfo W) {}
         );
      if (Overlap) {
        break;
      }
    }
    if (Overlap)
      break;
  }
  return Overlap;
}

class RegionsInfo {
public:

  void AddSingleRegion(const ForStmt* AST, Loop* IR, const Stmt* Parent,
    Function* F,
    ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
      mRegions[Parent].push_back({ AST, IR, Parent, F, VarInfo, IsHost});
  }
  
  void TryUnionParallelRegions() {
    for (auto item: mRegions) {
      Sort(item.second);
      TryUnionParallelRegions(item.first, item.second);
    }
  }

  void TryUnionActualRegions(DenseMap<Function*,
      FunctionAnalysisResult> FuncAnalysis ) {
    for (auto item : mRegions) {
      Sort(item.second);
      auto& FA = FuncAnalysis[item.second[0].GetFunction()];
      TryUnionActualRegions(item.first, item.second, std::get<0>(FA),
        std::get<1>(FA), std::get<2>(FA), std::get<3>(FA), std::get<4>(FA),
        std::get<5>(FA), std::get<6>(FA));
    }
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    for (auto& item : mRegions) {
      for (auto& Region : item.second) {
        Region.InsertPragmas(TfmCtx);
      }
    }
  }

  void Dump() {
    for (auto& item : mRegions) {
      dbgs() << "============\nParent: ";
      item.first->dump();
      for (auto& Region : item.second) {
        dbgs() << "--------------\n";
        Region.Dump();
        dbgs() << "--------------\n";
      }
      dbgs() << "============\n";
    }
  }

private:
  void Sort(std::vector<ActualRegion>& ActualRegions) {
    std::sort(ActualRegions.begin(), ActualRegions.end(),
      [](ActualRegion& R1, ActualRegion& R2) {
        return R1.GetLocStart() < R2.GetLocStart();
      });
  }

  /// Try union actual regions with parallel regions.
  /// From:
  ///   ....
  ///   actual1(A)
  ///   region1
  ///   get_actual(A)
  ///   actual2(B)
  ///   region2
  ///   get_actual2(B)
  ///   ....
  /// To:
  ///   ...
  ///   actual(A, B)
  ///   region
  ///   get_actual(A, B)
  ///   ...
  /// Works only if parallel regions have only one loop inside.
  void TryUnionParallelRegions(const Stmt* Parent,
      std::vector<ActualRegion>& Regions) {
    if (IsParallelRegionUnion)
      return;
    auto& AllChildren = Parent->children();
    std::vector<ActualRegion> UnionRegions;
    int LI = 0;
    auto tmpAR = ActualRegion(Parent, Regions[0].GetFunction());
    for (auto& CI = AllChildren.begin(), CE = AllChildren.end();
      CI != CE && LI < Regions.size(); ++CI) {
      auto Loop = Regions[LI].GetLoop();
      if (static_cast<Stmt const*>(Loop) == *CI) {
        tmpAR.AppendParallelRegion(Regions[LI++]);
      }
      else {
        if (!tmpAR.Empty()) {
          UnionRegions.push_back(tmpAR);
          tmpAR.Clear();
        }
      }
    }
    if (!tmpAR.Empty()) {
      UnionRegions.push_back(tmpAR);
    }
    mRegions[Parent] = UnionRegions;
    IsParallelRegionUnion = true;
  }

  void TryUnionActualRegions(const Stmt* Parent, 
      std::vector<ActualRegion>& Regions, DominatorTree* DT,
      PostDominatorTree* PDT, TargetLibraryInfo* TLI, AliasTree* AT,
      DIMemoryClientServerInfo* DIMInfo, const CanonicalLoopSet* CLS, 
      DFRegionInfo* RI) {
    auto CR = Regions.begin(), NR = Regions.begin();
    std::vector<ActualRegion> UnionRegions;
    auto tmpAR = ActualRegion(Parent, Regions[0].GetFunction());

    tmpAR.AppendActualRegion(*CR);
    for (++NR; NR != Regions.end(); ++CR, ++NR) {
      assert(CR->GetParent() == NR->GetParent() && CR->GetParent() == Parent);
      auto CPR = CR->GetRegion(0);
      auto NPR = NR->GetRegion(0);
      auto CL = CPR.GetFirstLoop();
      auto NL = NPR.GetFirstLoop();
      auto CIRL = CPR.GetIrLoop(CL);
      auto NIRL = NPR.GetIrLoop(NL);
      assert(CIRL != nullptr && NIRL != nullptr);
      assert(CIRL->getHeader() != nullptr && NIRL->getHeader() != nullptr);

      bool IsPostDom = PDT->dominates(NIRL->getHeader(), CIRL->getHeader());
      bool IsDom = DT->dominates(CIRL->getHeader(), NIRL->getHeader());
      if (!IsPostDom || !IsDom ||
          findMemoryOverlap(*CR, *NR, TLI, AT, DIMInfo, DT, CLS, RI)) {
          if (!tmpAR.Empty()) {
            UnionRegions.push_back(tmpAR);
            tmpAR.Clear();
          }
          tmpAR.AppendActualRegion(*NR);
      } else {
        tmpAR.AppendActualRegion(*NR);
      }
    }
    if (!tmpAR.Empty()) {
      UnionRegions.push_back(tmpAR);
    }
    mRegions[Parent] = UnionRegions;
  }
private:
  bool IsParallelRegionUnion{ false };
  DenseMap<const Stmt*, std::vector<ActualRegion>> mRegions;
};

/// This pass try to insert DVMH directives into a source code to obtain
/// a parallel program.
class ClangDVMHSMParallelization : public ClangSMParallelization {
public:
  static char ID;
  ClangDVMHSMParallelization() : ClangSMParallelization(ID) {
    initializeClangDVMHSMParallelizationPass(*PassRegistry::getPassRegistry());
  }

private:
  bool exploitParallelism(Loop& IR, const clang::ForStmt& AST,
    Function* F, const ClangSMParallelProvider& Provider,
    tsar::ClangDependenceAnalyzer& ASTDepInfo,
    tsar::TransformationContext& TfmCtx) override;

  void optimizeLevel(tsar::TransformationContext& TfmCtx) override;

  void finalize(tsar::TransformationContext& TfmCtx) override;

  RegionsInfo mRegionsInfo;

  DenseMap<Function *, FunctionAnalysisResult> mFuncAnalysis;

  TargetLibraryInfo* mTLI;
};

} // namespace

bool ClangDVMHSMParallelization::exploitParallelism(
    Loop &IR, const clang::ForStmt &AST,
    Function* F, const ClangSMParallelProvider &Provider,
    tsar::ClangDependenceAnalyzer &ASTRegionAnalysis,
    TransformationContext &TfmCtx) {

  auto& ASTCtx = TfmCtx.getContext();
  auto Parent = ASTCtx.getParents(AST)[0].get<Stmt>();
  auto &ASTDepInfo = ASTRegionAnalysis.getDependenceInfo();
  if (!ASTDepInfo.get<trait::FirstPrivate>().empty() ||
      !ASTDepInfo.get<trait::LastPrivate>().empty())
    return false;
  auto & PI = Provider.get<ParallelLoopPass>().getParallelLoopInfo();
  bool IsHost = PI[&IR].isHostOnly() || !ASTRegionAnalysis.evaluateDefUse();
  
  mRegionsInfo.AddSingleRegion(&AST, &IR, Parent, F, ASTDepInfo, IsHost);
  
  if (mFuncAnalysis.count(F) == 0) {
    auto DIAT = &Provider.get<DIEstimateMemoryPass>().getAliasTree();
    auto DIMInfo = new DIMemoryClientServerInfo(*this, *F, DIAT);
    assert(DIMInfo->isValid());

    mFuncAnalysis[F] = { &Provider.get<DominatorTreeWrapperPass>().getDomTree(),
      &Provider.get<PostDominatorTreeWrapperPass>().getPostDomTree(),
      &Provider.get<TargetLibraryInfoWrapperPass>().getTLI(),
      &Provider.get<EstimateMemoryPass>().getAliasTree(),
      DIMInfo,
      &Provider.get<CanonicalLoopPass>().getCanonicalLoopInfo(),
      &Provider.get<DFRegionInfoPass>().getRegionInfo()};
  }
  
  return true;
}

void ClangDVMHSMParallelization::optimizeLevel(
    tsar::TransformationContext& TfmCtx) {
  mRegionsInfo.TryUnionParallelRegions();
  mRegionsInfo.TryUnionActualRegions(mFuncAnalysis);
  mRegionsInfo.Dump();
}

void ClangDVMHSMParallelization::finalize(
  tsar::TransformationContext& TfmCtx) {
    mRegionsInfo.InsertPragmas(TfmCtx);
}

ModulePass *llvm::createClangDVMHSMParallelization() {
  return new ClangDVMHSMParallelization;
}

char ClangDVMHSMParallelization::ID = 0;
INITIALIZE_SHARED_PARALLELIZATION(ClangDVMHSMParallelization,
  "clang-dvmh-sm-parallel", "Shared Memory DVMH-based Parallelization (Clang)")
