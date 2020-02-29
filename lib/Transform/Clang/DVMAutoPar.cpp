//===-- DVMAutoPar.cpp - DVM Based Parallelization (Clang) -*- C++ -*-----===//
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
// This file implements a pass to perform DVM-based auto parallelization.
//
//===----------------------------------------------------------------------===//

#include "tsar/ADT/SpanningTreeRelation.h"
#include "tsar/Analysis/AnalysisServer.h"
#include "tsar/Analysis/Clang/CanonicalLoop.h"
#include "tsar/Analysis/Clang/DIMemoryMatcher.h"
#include "tsar/Analysis/Clang/LoopMatcher.h"
#include "tsar/Analysis/Clang/MemoryMatcher.h"
#include "tsar/Analysis/Clang/RegionDirectiveInfo.h"
#include "tsar/Analysis/DFRegionInfo.h"
#include "tsar/Analysis/Memory/ClonedDIMemoryMatcher.h"
#include "tsar/Analysis/Memory/DefinedMemory.h"
#include "tsar/Analysis/Memory/DIDependencyAnalysis.h"
#include "tsar/Analysis/Memory/DIEstimateMemory.h"
#include "tsar/Analysis/Memory/LiveMemory.h"
#include "tsar/Analysis/Memory/MemoryTraitUtils.h"
#include "tsar/Analysis/Memory/Passes.h"
#include "tsar/Analysis/Memory/ServerUtils.h"
#include "tsar/Analysis/Memory/EstimateMemory.h"
#include "tsar/Analysis/Parallel/ParallelLoop.h"
#include "tsar/Core/Query.h"
#include "tsar/Core/TransformationContext.h"
#include "tsar/Support/Clang/Diagnostic.h"
#include "tsar/Support/GlobalOptions.h"
#include "tsar/Support/PassAAProvider.h"
#include "tsar/Transform/Clang/Passes.h"
#include "tsar/Transform/IR/InterprocAttr.h"
#include <bcl/utility.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/CallGraphSCCPass.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Pass.h>
#include <algorithm>

using namespace llvm;
using namespace tsar;

#undef DEBUG_TYPE
#define DEBUG_TYPE "clang-dvm-parallel"

namespace llvm {
static void initializeClangDVMServerPass(PassRegistry &);
static void initializeClangDVMServerResponsePass(PassRegistry &);
}

namespace {
/// This provides access to function-level analysis results on server.
using ClangDVMServerProvider =
    FunctionPassAAProvider<DIEstimateMemoryPass, DIDependencyAnalysisPass>;

/// List of responses available from server (client may request corresponding
/// analysis, in case of provider all analysis related to a provider may
/// be requested separately).
using ClangDVMServerResponse = AnalysisResponsePass<
    GlobalsAAWrapperPass, DIMemoryTraitPoolWrapper, DIMemoryEnvironmentWrapper,
    GlobalDefinedMemoryWrapper, GlobalLiveMemoryWrapper,
    ClonedDIMemoryMatcherWrapper, ClangDVMServerProvider>;

/// This analysis server performs transformation-based analysis which is
/// necessary for DVM-based parallelization.
class ClangDVMServer final : public AnalysisServer {
public:
  static char ID;
  ClangDVMServer() : AnalysisServer(ID) {
    initializeClangDVMServerPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AnalysisServer::getAnalysisUsage(AU);
    ClientToServerMemory::getAnalysisUsage(AU);
    AU.addRequired<GlobalOptionsImmutableWrapper>();
  }

  void prepareToClone(Module &ClientM,
                      ValueToValueMapTy &ClientToServer) override {
    ClientToServerMemory::prepareToClone(ClientM, ClientToServer);
  }

  void initializeServer(Module &CM, Module &SM, ValueToValueMapTy &CToS,
                        legacy::PassManager &PM) override {
    auto &GO = getAnalysis<GlobalOptionsImmutableWrapper>();
    PM.add(createGlobalOptionsImmutableWrapper(&GO.getOptions()));
    PM.add(createGlobalDefinedMemoryStorage());
    PM.add(createGlobalLiveMemoryStorage());
    PM.add(createDIMemoryTraitPoolStorage());
    ClientToServerMemory::initializeServer(*this, CM, SM, CToS, PM);
  }

  void addServerPasses(Module &M, legacy::PassManager &PM) override {
    auto &GO = getAnalysis<GlobalOptionsImmutableWrapper>().getOptions();
    addImmutableAliasAnalysis(PM);
    addBeforeTfmAnalysis(PM);
    addAfterSROAAnalysis(GO, M.getDataLayout(), PM);
    addAfterLoopRotateAnalysis(PM);
    PM.add(createVerifierPass());
    PM.add(new ClangDVMServerResponse);
  }

  void prepareToClose(legacy::PassManager &PM) override {
    ClientToServerMemory::prepareToClose(PM);
  }
};

/// This provider access to function-level analysis results on client.
using ClangDVMParalleizationProvider =
    FunctionPassAAProvider<AnalysisSocketImmutableWrapper, LoopInfoWrapperPass,
                           ParallelLoopPass, CanonicalLoopPass, LoopMatcherPass,
                           DFRegionInfoPass, ClangDIMemoryMatcherPass, EstimateMemoryPass,
                           LiveMemoryPass, DominatorTreeWrapperPass>;

// Sorted list of variables (to print their in algoristic order).
using SortedVarListT = std::set<std::string, std::less<std::string>>;

// Lists of reduction variables.
using ReductionVarListT =
std::array<SortedVarListT, trait::DIReduction::RK_NumberOf>;

/// This pass try to insert DVM directives into a source code to obtain
/// a parallel program.
class ClangDVMParalleization : public ModulePass, private bcl::Uncopyable {
public:
  static char ID;
  ClangDVMParalleization() : ModulePass(ID) {
    initializeClangDVMParalleizationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override {
    mSkippedFuncs.clear();
    mRegions.clear();
    mTfmCtx = nullptr;
    mGlobalOpts = nullptr;
    mMemoryMatcher = nullptr;
    mGlobalsAA = nullptr;
    mSocket = nullptr;
  }

private:
  /// Initialize provider before on the fly passes will be run on client.
  void initializeProviderOnClient(Module &M);

  /// Initialize provider before on the fly passes will be run on server.
  void initializeProviderOnServer();

  TransformationContext *mTfmCtx = nullptr;
  const GlobalOptions *mGlobalOpts = nullptr;
  MemoryMatchInfo *mMemoryMatcher = nullptr;
  GlobalsAAResult * mGlobalsAA = nullptr;
  AnalysisSocket *mSocket = nullptr;
  DenseSet<Function *> mSkippedFuncs;
  SmallVector<const OptimizationRegion *, 4> mRegions;
};

/// This specifies additional passes which must be run on client.
class ClangDVMParallelizationInfo final : public PassGroupInfo {
  void addBeforePass(legacy::PassManager &Passes) const override {
    addImmutableAliasAnalysis(Passes);
    addInitialTransformations(Passes);
    Passes.add(createAnalysisSocketImmutableStorage());
    Passes.add(createDIMemoryTraitPoolStorage());
    Passes.add(createDIMemoryEnvironmentStorage());
    Passes.add(createDIEstimateMemoryPass());
    Passes.add(new ClangDVMServer);
    Passes.add(createAnalysisWaitServerPass());
    Passes.add(createMemoryMatcherPass());
  }
  void addAfterPass(legacy::PassManager &Passes) const override {
    Passes.add(createAnalysisReleaseServerPass());
    Passes.add(createAnalysisCloseConnectionPass());
  }
};

/// Return number of nested pointer-like types.
unsigned numberOfPointerTypes(const clang::Type *T) {
  if (auto PtrT = dyn_cast<clang::PointerType>(T))
    return numberOfPointerTypes(PtrT->getPointeeType().getTypePtr()) + 1;
  if (auto RefT = dyn_cast<clang::ReferenceType>(T))
    return numberOfPointerTypes(RefT->getPointeeType().getTypePtr()) + 1;
  if (auto ArrayT = dyn_cast<clang::ArrayType>(T))
    return numberOfPointerTypes(ArrayT->getElementType().getTypePtr());
  return 0;
}

/// Look up for locations referenced and declared in the scope.
struct VariableCollector
    : public clang::RecursiveASTVisitor<VariableCollector> {
  enum DeclSearch : uint8_t {
    /// Set if memory safely represent a local variable.
    CoincideLocal,
    /// Set if memory safely represent a local variable.
    CoincideGlobal,
    /// Set if memory does not represent the whole variable, however
    /// a corresponding variable can be used to describe a memory location.
    Derived,
    /// Set if corresponding variable does not exist.
    Invalid,
    /// Set if a found declaration is not explicitly mentioned in a loop body.
    /// For example, global variables may be used in called functions instead
    /// of a loop body.
    Implicit,
    /// Set if memory does not represent a variable. For example, it may
    /// represent a memory used in a function call or result of a function.
    Useless,
    /// Set if it is not known whether corresponding variable must exist or not.
    /// For example, a memory may represent some internal object which is not
    /// referenced in the original source code.
    Unknown,
  };
  
  static const clang::Type *getCanonicalUnqualifiedType(clang::VarDecl *VD) {
    return VD->getType()
        .getTypePtr()
        ->getCanonicalTypeUnqualified()
        ->getTypePtr();
  }

  /// Remember all referenced canonical declarations and compute number of
  /// estimate memory locations which should be built for this variable.
  bool VisitDeclRefExpr(clang::DeclRefExpr *DRE) {
    auto *ND = DRE->getFoundDecl();
    assert(ND && "Declaration must not be null!");
    if (isa<clang::VarDecl>(ND)) {
      auto *VD = cast<clang::VarDecl>(ND->getCanonicalDecl());
      if (!Induction)
        Induction = VD;
      auto T = getCanonicalUnqualifiedType(VD);
      CanonicalRefs.try_emplace(VD).first->second.resize(
        numberOfPointerTypes(T) + 1, nullptr);
    }
    return true;
  }

  /// Remember all canonical declarations declared inside the loop.
  bool VisitDeclStmt(clang::DeclStmt *DS) {
    for (auto *D : DS->decls())
      if (auto *Var = dyn_cast<clang::VarDecl>(D->getCanonicalDecl()))
        CanonicalLocals.insert(Var);
    return true;
  }

  /// Find declaration for a specified memory, remember memory if it safely
  /// represent a found variable or its part (update `CanonicalRefs` map).
  std::pair<clang::VarDecl *, DeclSearch>
  findDecl(const DIMemory &DIM,
           const ClangDIMemoryMatcherPass::DIMemoryMatcher &ASTToClient,
           const ClonedDIMemoryMatcher &ClientToServer) {
    auto *M = const_cast<DIMemory *>(&DIM);
    if (auto *DIEM = dyn_cast<DIEstimateMemory>(M)) {
      auto CSMemoryItr = ClientToServer.find<Clone>(DIEM);
      assert(CSMemoryItr != ClientToServer.end() &&
             "Metadata-level memory must exist on on client!");
      auto *DIVar =
          cast<DIEstimateMemory>(CSMemoryItr->get<Origin>())->getVariable();
      assert(DIVar && "Variable must not be null!");
      auto MatchItr = ASTToClient.find<MD>(DIVar);
      if (MatchItr == ASTToClient.end())
        return std::make_pair(nullptr, Invalid);
      auto ASTRefItr = CanonicalRefs.find(MatchItr->get<AST>());
      if (ASTRefItr == CanonicalRefs.end())
        return std::make_pair(MatchItr->get<AST>(), Implicit);
      if (DIEM->getExpression()->getNumElements() > 0) {
        auto *Expr = DIEM->getExpression();
        auto NumDeref = llvm::count(Expr->getElements(), dwarf::DW_OP_deref);
        auto *T = getCanonicalUnqualifiedType(ASTRefItr->first);
        // We want to be sure that current memory location describes all
        // possible memory locations which can be represented with a
        // corresponding variable and a specified number of its dereferences.
        // For example:
        // - <A,10> is sufficient to represent all memory defined by
        //   `int A[10]` (0 deref),
        // - <A,8> and <*A,?> are sufficient to represent all memory defined by
        //   `int (*A)[10]` (0 deref and 1 deref respectively).
        // - <A,8>, <*A,?>, <*A[?],?> are sufficient to represent all memory
        //   defined by `int **A` (0, 1 and 2 deref respectively).
        if (NumDeref < ASTRefItr->second.size() && !DIEM->isSized())
          if ((NumDeref == 1 && (NumDeref == Expr->getNumElements() ||
                                 Expr->isFragment() &&
                                     NumDeref == Expr->getNumElements() - 3)) ||
              (DIEM->isTemplate() && [](DIExpression *Expr) {
                // Now we check whether all offsets are zero. On success,
                // this means that all possible offsets are represented by
                // the template memory location DIEM.
                for (auto &Op : Expr->expr_ops())
                  switch (Op.getOp()) {
                  default:
                    llvm_unreachable("Unsupported kind of operand!");
                    return false;
                  case dwarf::DW_OP_deref:
                    break;
                  case dwarf::DW_OP_LLVM_fragment:
                  case dwarf::DW_OP_constu:
                  case dwarf::DW_OP_plus_uconst:
                  case dwarf::DW_OP_plus:
                  case dwarf::DW_OP_minus:
                    if (Op.getArg(0) == 0)
                      return false;
                  }
              }(Expr)))
            ASTRefItr->second[NumDeref] = DIEM;
        return std::make_pair(MatchItr->get<AST>(), Derived);
      }
      ASTRefItr->second.front() = DIEM;
      return std::make_pair(MatchItr->get<AST>(), isa<DILocalVariable>(DIVar)
                                                      ? CoincideLocal
                                                      : CoincideGlobal);
    }
    if (cast<DIUnknownMemory>(M)->isDistinct())
      return std::make_pair(nullptr, Unknown);
    return std::make_pair(nullptr, Useless);
  }

  /// Check whether it is possible to use high-level syntax to create copy for
  /// all memory locations in `TS` for each thread.
  ///
  /// On failure if `Error` not nullptr set it to the first variable which
  /// prevents localization (or to nullptr if variable not found).
  bool localize(DIAliasTrait &TS,
                const ClangDIMemoryMatcherPass::DIMemoryMatcher &ASTToClient,
                const ClonedDIMemoryMatcher &ClientToServer,
                SortedVarListT &VarNames, clang::VarDecl **Error = nullptr) {
    for (auto &T : TS)
      if (!localize(*T, *TS.getNode(), ASTToClient, ClientToServer, VarNames))
        return false;
    return true;
  }

  /// Check whether it is possible to use high-level syntax to create copy of a
  /// specified memory `T` for each thread.
  ///
  /// On success to create a local copy of a memory source-level variable
  /// should be mentioned in a clauses like private or reduction.
  /// This variable will be stored in a list of variables `VarNames`.
  /// \attention  This method does not check whether it is valid to create a
  /// such copy, for example global variables must be checked later.
  /// Localized global variables breaks relation with original global variables.
  /// And program may become invalid if such variables are used in calls inside
  /// the loop body.
  /// \post On failure if `Error` not nullptr set it to the first variable which
  /// prevents localization (or to nullptr if variable not found).
  bool localize(DIMemoryTrait &T, const DIAliasNode &DIN,
                const ClangDIMemoryMatcherPass::DIMemoryMatcher &ASTToClient,
                const ClonedDIMemoryMatcher &ClientToServer,
                SortedVarListT &VarNames, clang::VarDecl **Error = nullptr) {
    auto Search = findDecl(*T.getMemory(), ASTToClient, ClientToServer);
    if (Search.second == VariableCollector::CoincideLocal) {
      // Do no specify traits for variables declared in a loop body
      // these variables are private by default. Moreover, these variables are
      // not visible outside the loop and could not be mentioned in clauses
      // before loop.
      if (!CanonicalLocals.count(Search.first))
        VarNames.insert(Search.first->getName());
    } else if (Search.second == VariableCollector::CoincideGlobal) {
      VarNames.insert(Search.first->getName());
      GlobalRefs.try_emplace(const_cast<DIAliasNode *>(&DIN), Search.first);
    } else if (Search.second != VariableCollector::Unknown) {
      if (Error)
        *Error = Search.first;
      return false;
    }
    return true;
  }

  clang::VarDecl * Induction = nullptr;
  DenseMap<clang::VarDecl *, SmallVector<DIEstimateMemory *, 2>> CanonicalRefs;
  DenseSet<clang::VarDecl *> CanonicalLocals;
  /// Map from alias node which contains global memory to one of global
  /// variables which represents this memory.
  DenseMap<DIAliasNode *, clang::VarDecl *> GlobalRefs;
};

} // namespace

void ClangDVMParalleization::initializeProviderOnClient(Module &M) {
  ClangDVMParalleizationProvider::initialize<GlobalOptionsImmutableWrapper>(
      [this](GlobalOptionsImmutableWrapper &Wrapper) {
        Wrapper.setOptions(mGlobalOpts);
      });
  ClangDVMParalleizationProvider::initialize<AnalysisSocketImmutableWrapper>(
      [this](AnalysisSocketImmutableWrapper &Wrapper) {
        Wrapper.set(*mSocket);
      });
  ClangDVMParalleizationProvider::initialize<TransformationEnginePass>(
      [this, &M](TransformationEnginePass &Wrapper) {
        Wrapper.setContext(M, mTfmCtx);
      });
  ClangDVMParalleizationProvider::initialize<MemoryMatcherImmutableWrapper>(
      [this](MemoryMatcherImmutableWrapper &Wrapper) {
        Wrapper.set(*mMemoryMatcher);
      });
  ClangDVMParalleizationProvider::initialize<
      GlobalsAAResultImmutableWrapper>(
      [this](GlobalsAAResultImmutableWrapper &Wrapper) {
        Wrapper.set(*mGlobalsAA);
      });
}

void ClangDVMParalleization::initializeProviderOnServer() {
  ClangDVMServerProvider::initialize<GlobalOptionsImmutableWrapper>(
      [this](GlobalOptionsImmutableWrapper &Wrapper) {
        Wrapper.setOptions(mGlobalOpts);
      });
  auto R = mSocket->getAnalysis<GlobalsAAWrapperPass,
      DIMemoryEnvironmentWrapper, DIMemoryTraitPoolWrapper,
      GlobalDefinedMemoryWrapper, GlobalLiveMemoryWrapper>();
  assert(R && "Immutable passes must be available on server!");
  auto *DIMEnvServer = R->value<DIMemoryEnvironmentWrapper *>();
  ClangDVMServerProvider::initialize<DIMemoryEnvironmentWrapper>(
      [DIMEnvServer](DIMemoryEnvironmentWrapper &Wrapper) {
        Wrapper.set(**DIMEnvServer);
      });
  auto *DIMTraitPoolServer = R->value<DIMemoryTraitPoolWrapper *>();
  ClangDVMServerProvider::initialize<DIMemoryTraitPoolWrapper>(
      [DIMTraitPoolServer](DIMemoryTraitPoolWrapper &Wrapper) {
        Wrapper.set(**DIMTraitPoolServer);
      });
  auto &GlobalsAAServer = R->value<GlobalsAAWrapperPass *>()->getResult();
  ClangDVMServerProvider::initialize<GlobalsAAResultImmutableWrapper>(
      [&GlobalsAAServer](GlobalsAAResultImmutableWrapper &Wrapper) {
        Wrapper.set(GlobalsAAServer);
      });
  auto *GlobalDefUseServer = R->value<GlobalDefinedMemoryWrapper *>();
  ClangDVMServerProvider::initialize<GlobalDefinedMemoryWrapper>(
    [GlobalDefUseServer](GlobalDefinedMemoryWrapper &Wrapper) {
      Wrapper.set(**GlobalDefUseServer);
    });
  auto *GlobalLiveMemoryServer = R->value<GlobalLiveMemoryWrapper *>();
  ClangDVMServerProvider::initialize<GlobalLiveMemoryWrapper>(
    [GlobalLiveMemoryServer](GlobalLiveMemoryWrapper &Wrapper) {
      Wrapper.set(**GlobalLiveMemoryServer);
    });
}

bool ClangDVMParalleization::runOnModule(Module &M) {
  releaseMemory();
  mTfmCtx = getAnalysis<TransformationEnginePass>().getContext(M);
  if (!mTfmCtx || !mTfmCtx->hasInstance()) {
    M.getContext().emitError("can not transform sources"
                             ": transformation context is not available");
    return false;
  }

  mSocket = &getAnalysis<AnalysisSocketImmutableWrapper>().get();
  mGlobalOpts = &getAnalysis<GlobalOptionsImmutableWrapper>().getOptions();
  mMemoryMatcher = &getAnalysis<MemoryMatcherImmutableWrapper>().get();
  mGlobalsAA = &getAnalysis<GlobalsAAWrapperPass>().getResult();
  initializeProviderOnClient(M);
  initializeProviderOnServer();
  auto &RegionInfo = getAnalysis<ClangRegionCollector>().getRegionInfo();
  if (mGlobalOpts->OptRegions.empty()) {
    transform(RegionInfo, std::back_inserter(mRegions),
              [](const OptimizationRegion &R) { return &R; });
  } else {
    for (auto &Name : mGlobalOpts->OptRegions)
      if (auto *R = RegionInfo.get(Name))
        mRegions.push_back(R);
      else
        toDiag(mTfmCtx->getContext().getDiagnostics(),
               clang::diag::warn_region_not_found) << Name;
  }
  auto &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    if (I->size() > 1)
      continue;
    auto *F = I->front()->getFunction();
    if (!F || F->isIntrinsic() || F->isDeclaration() ||
        hasFnAttr(*F, AttrKind::LibFunc) || mSkippedFuncs.count(F))
      continue;
    if (!mRegions.empty() && std::all_of(mRegions.begin(), mRegions.end(),
                                         [F](const OptimizationRegion *R) {
                                           return R->contain(*F) ==
                                                  OptimizationRegion::CS_No;
                                         }))
      continue;
    dbgs() << "[DVM PARALLEL]: process function " << F->getName()
                      << "\n";
    auto &Provider = getAnalysis<ClangDVMParalleizationProvider>(*F);    
    auto &LI = Provider.get<LoopInfoWrapperPass>().getLoopInfo();
    auto& LM = Provider.get<LiveMemoryPass>().getLiveInfo();
    auto& RegInfo = Provider.get<DFRegionInfoPass>().getRegionInfo();
    auto& PL = Provider.get<ParallelLoopPass>().getParallelLoopInfo();
    auto& AT = Provider.get<EstimateMemoryPass>().getAliasTree();
    auto& DT = Provider.get<DominatorTreeWrapperPass>().getDomTree();
    auto RF =
      mSocket->getAnalysis<DIEstimateMemoryPass, DIDependencyAnalysisPass>(*F);
    assert(RF && "Dependence analysis must be available for a parallel loop!");
    auto& DIAT = RF->value<DIEstimateMemoryPass*>()->getAliasTree();
    auto& DL = F->getParent()->getDataLayout();

    for (auto LIter = LI.begin(), End = LI.end(); LIter != End; LIter++) {
      llvm::Loop* L = *LIter;
      if (PL.count(L)) {
        dbgs() << (L)->getName() << "\n";
        auto& LMR = LM[RegInfo.getRegionFor(L)].get()->getOut();
        dbgs() << "Live Variables: \n";
        for (auto& MLR : LMR) {
          auto* EM = AT.find(MLR);
          assert(EM && "Estimate memory must be presented in alias tree!");
          auto RawDIM = getRawDIMemoryIfExists(
            *EM->getTopLevelParent(), F->getContext(), DL, DT);
          assert(RawDIM && "Unknown raw memory!");
          assert(DIAT.find(*RawDIM) != DIAT.memory_end() &&
            "Memory must exist in alias tree!");
          auto& DIEM = cast<DIEstimateMemory>(*DIAT.find(*RawDIM));
          auto Var = DIEM.getVariable();
          dbgs() << "\t" << Var->getName() << "\n";
        }
      }
    }
  }
  return false;
}

ModulePass *llvm::createClangDVMParallelization() {
  return new ClangDVMParalleization;
}

void ClangDVMParalleization::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ClangDVMParalleizationProvider>();
  AU.addRequired<AnalysisSocketImmutableWrapper>();
  AU.addRequired<TransformationEnginePass>();
  AU.addRequired<MemoryMatcherImmutableWrapper>();
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<GlobalOptionsImmutableWrapper>();
  AU.addRequired<GlobalsAAWrapperPass>();
  AU.addRequired<ClangRegionCollector>();
  AU.setPreservesAll();
}

INITIALIZE_PROVIDER(ClangDVMServerProvider, "clang-dvm-server-provider",
                    "DVM Based Parallelization (Clang, Server, Provider)")

template <> char ClangDVMServerResponse::ID = 0;
INITIALIZE_PASS(ClangDVMServerResponse, "clang-dvm-parallel-response",
                "DVM Based Parallelization (Clang, Server, Response)", true,
                false)

char ClangDVMServer::ID = 0;
INITIALIZE_PASS(ClangDVMServer, "clang-dvm-parallel-server",
                "DVM Based Parallelization (Clang, Server)", false, false)

INITIALIZE_PROVIDER(ClangDVMParalleizationProvider,
                    "clang-dvm-parallel-provider",
                    "DVM Based Parallelization (Clang, Provider)")

char ClangDVMParalleization::ID = 0;
INITIALIZE_PASS_IN_GROUP_BEGIN(ClangDVMParalleization,
                               "clang-dvm-parallel",
                               "DVM Based Parallelization (Clang)", false,
                               false,
                               TransformationQueryManager::getPassRegistry())
INITIALIZE_PASS_IN_GROUP_INFO(ClangDVMParallelizationInfo)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DFRegionInfoPass)
INITIALIZE_PASS_DEPENDENCY(DIEstimateMemoryPass)
INITIALIZE_PASS_DEPENDENCY(DIDependencyAnalysisPass)
INITIALIZE_PASS_DEPENDENCY(ClangDVMParalleizationProvider)
INITIALIZE_PASS_DEPENDENCY(TransformationEnginePass)
INITIALIZE_PASS_DEPENDENCY(MemoryMatcherImmutableWrapper)
INITIALIZE_PASS_DEPENDENCY(GlobalDefinedMemoryWrapper)
INITIALIZE_PASS_DEPENDENCY(GlobalLiveMemoryWrapper)
INITIALIZE_PASS_DEPENDENCY(GlobalOptionsImmutableWrapper)
INITIALIZE_PASS_DEPENDENCY(DIMemoryEnvironmentWrapper)
INITIALIZE_PASS_DEPENDENCY(DIMemoryTraitPoolWrapper)
INITIALIZE_PASS_DEPENDENCY(ClangDVMServerProvider)
INITIALIZE_PASS_DEPENDENCY(ClonedDIMemoryMatcherWrapper)
INITIALIZE_PASS_DEPENDENCY(ClangDVMServerResponse)
INITIALIZE_PASS_DEPENDENCY(ParallelLoopPass)
INITIALIZE_PASS_DEPENDENCY(EstimateMemoryPass)
INITIALIZE_PASS_DEPENDENCY(LiveMemoryPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CanonicalLoopPass)
INITIALIZE_PASS_DEPENDENCY(ClangRegionCollector)
INITIALIZE_PASS_IN_GROUP_END(ClangDVMParalleization, "clang-dvm-parallel",
                             "DVM Based Parallelization (Clang)", false,
                             false,
                             TransformationQueryManager::getPassRegistry())
