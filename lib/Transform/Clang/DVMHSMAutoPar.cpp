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
#include "tsar/Analysis/Parallel/Passes.h"
#include "tsar/Analysis/Parallel/ParallelLoop.h"
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

void addVarList(const ClangDependenceAnalyzer::SortedVarListT& VarInfoList,
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
void addVarList(const ClangDependenceAnalyzer::ReductionVarListT& VarInfoList,
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

  ParallelRegion(const ForStmt* AST, const Loop* IR, const Stmt* Parent,
      ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
    mLoops.push_back(AST);
    mParent = Parent;
    mIrLoops[AST] = IR;
    mVarInfo[AST] = VarInfo;
    mHostInfo[AST] = IsHost;
  }

  SourceLocation GetLocStart() {
    if (!mIsSorted)
      Sort();
    return mLoops[0]->getLocStart();
  }

  SourceLocation GetLocEnd() {
    if (!mIsSorted)
      Sort();
    return mLoops[mLoops.size() - 1]->getLocEnd();
  }

  const ForStmt* GetLoop() const {
    assert(mLoops.size() == 1);
    return mLoops[0];
  }

  std::vector <const ForStmt*>& GetLoops() {
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

  const Loop* GetIrLoop(const ForStmt* AST) {
    assert(mIrLoops.count(AST));
    return mIrLoops[AST];
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
    assert(!mHostInfo.count(AST) && !mIrLoops.count(AST) && !mVarInfo.count(AST));
    mLoops.push_back(AST);
    mHostInfo[AST] = Region.GetHostInfo(AST);
    mIrLoops[AST] = Region.GetIrLoop(AST);
    mVarInfo[AST] = Region.GetVarInfo(AST);
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    InsertAllPragmaParallel(TfmCtx);
    InsertPragmaRegion(TfmCtx);
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
    auto& ASTCtx = TfmCtx.getContext();
    Token SemiTok;
    auto InsertLoc = (!getRawTokenAfter(GetLocEnd(),
      ASTCtx.getSourceManager(), ASTCtx.getLangOpts(), SemiTok)
      && SemiTok.is(tok::semi))
      ? SemiTok.getLocation() : GetLocEnd();
    Rewriter.InsertTextAfterToken(InsertLoc, "}");
  }

  void Sort() {
    std::sort(mLoops.begin(), mLoops.end(),
      [](const ForStmt* F1, const ForStmt* F2) {
        return F1->getLocStart() < F2->getLocEnd();
      });
    mIsSorted = true;
  }

private:
  bool mIsSorted{ false };
  // Loops in region
  std::vector <const ForStmt*> mLoops;

  DenseMap<const ForStmt*, bool> mHostInfo;

  // IR representation of loop
  DenseMap<const ForStmt*, const Loop*> mIrLoops;

  // Var info for pragmas
  DenseMap<const ForStmt*,
    ClangDependenceAnalyzer::ASTRegionTraitInfo> mVarInfo;

  const Stmt* mParent;
};

/// This class stores information about regions in one #pragma dvm actual/get_actual
/// actual (...)
/// region 1
/// region 2
/// get_actual(...)
class ActualRegion {
public:

  ActualRegion(const ForStmt* AST, const Loop* IR, const Stmt* Parent,
    ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
    mRegions.push_back({ AST, IR, Parent, VarInfo, IsHost});
    mParent = Parent;
  }

  ActualRegion(const Stmt* Parent) {
    mParent = Parent;
  }

  SourceLocation GetLocStart() {
    if (!mIsSorted)
      Sort();
    return mRegions[0].GetLocStart();
  }

  SourceLocation GetLocEnd() {
    if (!mIsSorted)
      Sort();
    return mRegions[mRegions.size() - 1].GetLocEnd();
  }

  const ForStmt* GetLoop() const {
    assert(mRegions.size() == 1);
    return mRegions[0].GetLoop();
  }

  ParallelRegion GetParallelRegion() const {
    assert(mRegions.size() == 1);
    return mRegions[0];
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
    mIsSorted = false;
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    for (auto& Region : mRegions) {
      Region.InsertPragmas(TfmCtx);
    }
    InsertPragmaActual(TfmCtx);
    InsertPragmaGetActual(TfmCtx);
  }

private:
  // TODO: extend this function for several regions in actual region
  void InsertPragmaActual(TransformationContext& TfmCtx) {
    SmallString<128> DVMHActual;
    if (!mRegions[0].GetUnionHostInfo()) {
      SortedVarListT UnionVarSet;
      unionVarSets<trait::ReadOccurred>(mRegions[0].GetLoops() , mRegions[0].GetVarInfo(), UnionVarSet);
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
      unionVarSets<trait::WriteOccurred>(mRegions[0].GetLoops(), mRegions[0].GetVarInfo(), UnionVarSet);
      if (!UnionVarSet.empty()) {
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
    mIsSorted = true;
  }

private:
  bool mIsSorted{ false };

  //Regions in actual regions
  std::vector<ParallelRegion> mRegions;

  const Stmt* mParent;
};

class RegionsInfo {
public:

  void AddSingleRegion(const ForStmt* AST, const Loop* IR, const Stmt* Parent,
    ClangDependenceAnalyzer::ASTRegionTraitInfo VarInfo, bool IsHost) {
      mRegions[Parent].push_back({ AST, IR, Parent, VarInfo, IsHost});
  }
  
  void TryUnionParallelRegions() {
    for (auto item: mRegions) {
      Sort(item.second);
      TryUnionParallelRegions(item.first, item.second);
    }
  }

  void InsertPragmas(TransformationContext& TfmCtx) {
    for (auto& item : mRegions) {
      for (auto& Region : item.second) {
        Region.InsertPragmas(TfmCtx);
      }
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
  ///   regin1
  ///   get_actual(A)
  ///   actual2(B)
  ///   regin2
  ///   get_actual2(B)
  ///   ....
  /// To:
  ///   ...
  ///   actual(A, B)
  ///   regin
  ///   get_actual(A, B)
  ///   ...
  void TryUnionParallelRegions(const Stmt* Parent, std::vector<ActualRegion>& Regions) {
    auto& AllChildren = Parent->children();
    std::vector<ActualRegion> UnionRegions;
    int LI = 0;
    auto tmpAR = ActualRegion(Parent);
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
  }

private:

  DenseMap<const Stmt*, std::vector<ActualRegion>> mRegions;
};

/// This pass try to insert OpenMP directives into a source code to obtain
/// a parallel program.
class ClangDVMHSMParallelization : public ClangSMParallelization {
public:
  static char ID;
  ClangDVMHSMParallelization() : ClangSMParallelization(ID) {
    initializeClangDVMHSMParallelizationPass(*PassRegistry::getPassRegistry());
  }
  // using ClangToIR = DenseMap<const ForStmt *, const Loop*>;
  // 
  // using ForVarInfoT = DenseMap<const ForStmt *,
  //   std::pair<ClangDependenceAnalyzer::ASTRegionTraitInfo, bool>>;
  // 
  // using ParentChildInfoT = DenseMap<const Stmt *,
  //   std::pair<std::vector<const ForStmt *>, bool>>;

private:
  bool exploitParallelism(const Loop &IR, const clang::ForStmt &AST,
    const ClangSMParallelProvider &Provider,
    tsar::ClangDependenceAnalyzer &ASTDepInfo,
    TransformationContext &TfmCtx) override;

  void optimizeLevel(tsar::TransformationContext& TfmCtx) override;

  RegionsInfo mRegionsInfo;

  // ClangToIR mCLangToIR;
  // ForVarInfoT mForVarInfo;
  // ParentChildInfoT mParentChildInfo;

  PostDominatorTree* mPDT;
  DominatorTree* mDT;
  TargetLibraryInfo* mTLI;

  bool first{ true };
};
// using ClangToIR = ClangDVMHSMParallelization::ClangToIR;
// using ForVarInfoT = ClangDVMHSMParallelization::ForVarInfoT;
// using ParentChildInfoT = ClangDVMHSMParallelization::ParentChildInfoT;

// using UnionRegionsT = std::vector<std::vector<ForStmt const*>>;
// using UnionActualsT = std::vector<std::vector<ForStmt const*>>;




// bool findMemoryOverlap(std::vector<const ForStmt*> R1, std::vector<const ForStmt*> R2, TargetLibraryInfo* TLI) {
//   for_each_memory(*cast<Instruction>(CallRecord.first), TLI,
//     [Callee, &CallLiveOut](Instruction& I, MemoryLocation&& Loc,
//       unsigned Idx, AccessInfo, AccessInfo) {
//         auto OverlapItr = CallLiveOut.findOverlappedWith(Loc);
//         if (OverlapItr == CallLiveOut.end())
//           return;
//         auto* Arg = Callee->arg_begin() + Idx;
//         CallLiveOut.insert(MemoryLocationRange(Arg, 0, Loc.Size));
//     },
//     [](Instruction&, AccessInfo, AccessInfo) {});
//   return true;
// }

// void tryUnionActuals(const UnionRegionsT& UR,
//     TransformationContext& TfmCtx, UnionActualsT& UA, DominatorTree* DT, PostDominatorTree* PDT, ClangToIR& CLIR) {
//   auto& ASTCtx = TfmCtx.getContext();
//   for (auto CR = UR.begin(); CR != UR.end(); ++CR) {
//     auto& NR = CR++;
//     CR--;
//     if (NR != UR.end()) {
//       auto CL = (*CR)[0];
//       auto NL = (*CR)[0];
//       auto CP = ASTCtx.getParents(*CL)[0].get<Stmt>();
//       auto NP = ASTCtx.getParents(*NL)[0].get<Stmt>();
//       if (CP == NP) {
//         if (DT->dominates(CLIR[CL]->getHeader(), CLIR[NL]->getHeader()) &&
//           PDT->dominates(CLIR[NL]->getHeader(), CLIR[CL]->getHeader())) {
// 
//         }
//       }
//     }
//   }
// }
} // namespace

bool ClangDVMHSMParallelization::exploitParallelism(
    const Loop &IR, const clang::ForStmt &AST,
    const ClangSMParallelProvider &Provider,
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
  
  mRegionsInfo.AddSingleRegion(&AST, &IR, Parent, ASTDepInfo, IsHost);
  
  if (mPDT == nullptr) {
    // mPDT = &Provider.get<PostDominatorTreeWrapperPass>().getPostDomTree();
  }
  if (mDT == nullptr) {
    // mDT = &Provider.get<DominatorTreeWrapperPass>().getDomTree();
  }
  
  return true;
}

void ClangDVMHSMParallelization::optimizeLevel(
    tsar::TransformationContext& TfmCtx) {
  // UnionRegionsT UnionRegions;
  // tryUnionRegions(mParentChildInfo, UnionRegions);
  // for (auto& UR : UnionRegions) {
  //   auto& FL = UR[0];
  //   auto& LL = UR[UR.size() - 1];
  //   bool IsHost = mForVarInfo[FL].second;
  //   SmallString<128> ParallelFor("#pragma dvm parallel (1)");
  //   SortedVarListT UnionPrivateVarSet;
  //   unionVarSets<trait::Private>(UR, mForVarInfo, UnionPrivateVarSet);
  //   if (!UnionPrivateVarSet.empty()) {
  //     ParallelFor += " private";
  //     addVarList(UnionPrivateVarSet, ParallelFor);
  //   }
  //   ReductionVarListT UnionRedVarSet;
  //   unionVarRedSets(UR, mForVarInfo, UnionRedVarSet);
  //   addVarList(UnionRedVarSet, ParallelFor);
  //   ParallelFor += '\n';
  //   SmallString<128> DVMHRegion("#pragma dvm region");
  //   SmallString<128> DVMHActual, DVMHGetActual;
  //   if (IsHost) {
  //     DVMHRegion += " targets(HOST)";
  //   }
  //   else {
  //     SortedVarListT UnionVarSet;
  //     unionVarSets<trait::ReadOccurred>(UR, mForVarInfo, UnionVarSet);
  //     if (!UnionVarSet.empty()) {
  //       DVMHActual += "#pragma dvm actual";
  //       addVarList(UnionVarSet, DVMHActual);
  //       DVMHRegion += " in";
  //       addVarList(UnionVarSet, DVMHRegion);
  //       DVMHActual += '\n';
  //       UnionVarSet.clear();
  //     }
  //     unionVarSets<trait::WriteOccurred>(UR, mForVarInfo, UnionVarSet);
  //     if (!UnionVarSet.empty()) {
  //       DVMHGetActual += "#pragma dvm get_actual";
  //       addVarList(UnionVarSet, DVMHGetActual);
  //       DVMHRegion += " out";
  //       addVarList(UnionVarSet, DVMHRegion);
  //       DVMHGetActual += '\n';
  //       UnionVarSet.clear();
  //     }
  //     if (!UnionPrivateVarSet.empty()) {
  //       DVMHRegion += " local";
  //       addVarList(UnionPrivateVarSet, DVMHRegion);
  //     }
  //   }
  //   DVMHRegion += "\n{\n";
  // 
  //   // Add directives to the source code.
  //   auto& Rewriter = TfmCtx.getRewriter();
  //   Rewriter.InsertTextBefore(FL->getLocStart(), ParallelFor);
  //   Rewriter.InsertTextBefore(FL->getLocStart(), DVMHRegion);
  //   if (!DVMHActual.empty())
  //     Rewriter.InsertTextBefore(FL->getLocStart(), DVMHActual);
  //   auto& ASTCtx = TfmCtx.getContext();
  //   Token SemiTok;
  //   auto InsertLoc = (!getRawTokenAfter(LL->getLocEnd(),
  //     ASTCtx.getSourceManager(), ASTCtx.getLangOpts(), SemiTok)
  //     && SemiTok.is(tok::semi))
  //     ? SemiTok.getLocation() : LL->getLocEnd();
  //   Rewriter.InsertTextAfterToken(InsertLoc, "}");
  //   if (!DVMHGetActual.empty()) {
  //     Rewriter.InsertTextAfterToken(InsertLoc, "\n");
  //     Rewriter.InsertTextAfterToken(InsertLoc, DVMHGetActual);
  //   }
  // }
  if (first) {
    mRegionsInfo.TryUnionParallelRegions();
    mRegionsInfo.InsertPragmas(TfmCtx);
    first = false;
  }
}

ModulePass *llvm::createClangDVMHSMParallelization() {
  return new ClangDVMHSMParallelization;
}

char ClangDVMHSMParallelization::ID = 0;
INITIALIZE_SHARED_PARALLELIZATION(ClangDVMHSMParallelization,
  "clang-dvmh-sm-parallel", "Shared Memory DVMH-based Parallelization (Clang)")
