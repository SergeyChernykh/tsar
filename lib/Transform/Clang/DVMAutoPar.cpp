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

#include "tsar/Analysis/AnalysisServer.h"
#include "tsar/Analysis/Clang/CanonicalLoop.h"
#include "tsar/Analysis/Clang/MemoryMatcher.h"
#include "tsar/Analysis/Clang/RegionDirectiveInfo.h"
#include "tsar/Analysis/DFRegionInfo.h"
#include "tsar/Analysis/Memory/DIDependencyAnalysis.h"
#include "tsar/Analysis/Memory/Passes.h"
#include "tsar/Analysis/Memory/ServerUtils.h"
#include "tsar/Analysis/Memory/EstimateMemory.h"
#include "tsar/Analysis/Parallel/ParallelLoop.h"
#include "tsar/Analysis/Parallel/LoopDefLiveVarInfo.h"
#include "tsar/Core/Query.h"
#include "tsar/Core/TransformationContext.h"
#include "tsar/Support/Clang/Diagnostic.h"
#include "tsar/Support/GlobalOptions.h"
#include "tsar/Support/PassAAProvider.h"
#include "tsar/Transform/Clang/Passes.h"
#include "tsar/Transform/IR/InterprocAttr.h"
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/CallGraphSCCPass.h>
#include <llvm/IR/Verifier.h>

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
    ClangDVMServerProvider, LoopDefLiveVarInfoPass>;

  /// This analysis server performs transformation-based analysis which is
  /// necessary for DVM-based parallelization.
  class ClangDVMServer final : public AnalysisServer {
  public:
    static char ID;
    ClangDVMServer() : AnalysisServer(ID) {
      initializeClangDVMServerPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage& AU) const override {
      AnalysisServer::getAnalysisUsage(AU);
      ClientToServerMemory::getAnalysisUsage(AU);
      AU.addRequired<GlobalOptionsImmutableWrapper>();
    }

    void prepareToClone(Module& ClientM,
      ValueToValueMapTy& ClientToServer) override {
      ClientToServerMemory::prepareToClone(ClientM, ClientToServer);
    }

    void initializeServer(Module& CM, Module& SM, ValueToValueMapTy& CToS,
      legacy::PassManager& PM) override {
      auto& GO = getAnalysis<GlobalOptionsImmutableWrapper>();
      PM.add(createGlobalOptionsImmutableWrapper(&GO.getOptions()));
      PM.add(createDIMemoryTraitPoolStorage());
      ClientToServerMemory::initializeServer(*this, CM, SM, CToS, PM);
    }

    void addServerPasses(Module& M, legacy::PassManager& PM) override {
      auto& GO = getAnalysis<GlobalOptionsImmutableWrapper>().getOptions();
      addImmutableAliasAnalysis(PM);
      addBeforeTfmAnalysis(PM);
      addAfterSROAAnalysis(GO, M.getDataLayout(), PM);
      addAfterLoopRotateAnalysis(PM);
      PM.add(createVerifierPass());
      PM.add(new ClangDVMServerResponse);
    }

    void prepareToClose(legacy::PassManager& PM) override {
      ClientToServerMemory::prepareToClose(PM);
    }
  };

  /// This provider access to function-level analysis results on client.
  using ClangDVMParalleizationProvider =
    FunctionPassAAProvider<AnalysisSocketImmutableWrapper,
    ParallelLoopPass, CanonicalLoopPass, DFRegionInfoPass>;


  /// This pass try to insert DVM directives into a source code to obtain
  /// a parallel program.
  class ClangDVMParalleization : public ModulePass, private bcl::Uncopyable {
  public:

    static char ID;
    ClangDVMParalleization() : ModulePass(ID) {
      initializeClangDVMParalleizationPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module& M) override;
    void getAnalysisUsage(AnalysisUsage& AU) const override;

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
    void initializeProviderOnClient(Module& M);

    /// Initialize provider before on the fly passes will be run on server.
    void initializeProviderOnServer();

    void findLoopDefLiveInfo(Function* F, ClangDVMParalleizationProvider& P,
      clang::Rewriter& Rewriter);

    SmallString<128>& buildPragma(SmallString<128> Prefix,
      llvm::DenseSet<DIVariable*> Vars);

    TransformationContext* mTfmCtx = nullptr;
    const GlobalOptions* mGlobalOpts = nullptr;
    MemoryMatchInfo* mMemoryMatcher = nullptr;
    GlobalsAAResult* mGlobalsAA = nullptr;
    AnalysisSocket* mSocket = nullptr;
    DenseSet<Function*> mSkippedFuncs;
    SmallVector<const OptimizationRegion*, 4> mRegions;
  };

  /// This specifies additional passes which must be run on client.
  class ClangDVMParallelizationInfo final : public PassGroupInfo {
    void addBeforePass(legacy::PassManager& Passes) const override {
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
    void addAfterPass(legacy::PassManager& Passes) const override {
      Passes.add(createAnalysisReleaseServerPass());
      Passes.add(createAnalysisCloseConnectionPass());
    }
  };
}
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
      DIMemoryEnvironmentWrapper, DIMemoryTraitPoolWrapper>();
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
}

void ClangDVMParalleization::findLoopDefLiveInfo(Function* F,
  ClangDVMParalleizationProvider& P, clang::Rewriter& Rewriter) {
  auto RF =
    mSocket->getAnalysis<LoopDefLiveVarInfoPass>(*F);
  assert(RF && "Dependence analysis must be available for a parallel loop!");
  auto RM = mSocket->getAnalysis<AnalysisClientServerMatcherWrapper>();
  assert(RM && "Client to server IR-matcher must be available!");
  auto& ClientToServer = **RM->value<AnalysisClientServerMatcherWrapper*>();
  auto& LDLI = RF->value<LoopDefLiveVarInfoPass*>()->getLoopDefLiveVarInfoInfo();
  auto& PL = P.get<ParallelLoopPass>().getParallelLoopInfo();
  auto& CL = P.get<CanonicalLoopPass>().getCanonicalLoopInfo();
  auto& RI = P.get<DFRegionInfoPass>().getRegionInfo();

  for (auto& L : PL) {
    assert(L->getLoopID() && "ID must be available for a parallel loop!");
    auto ServerLoopID = cast<MDNode>(*ClientToServer.getMappedMD(L->getLoopID()));
    auto LDLILoop = LDLI[ServerLoopID];
    SmallString<128> Actual("#pragma dvm actual(");
    SmallString<128> GetActual("\n#pragma dvm get_actual(");
    auto CanonicalItr = CL.find_as(RI.getRegionFor(L));
    auto* ForStmt = (**CanonicalItr).getASTLoop();
    assert(ForStmt && "Source-level loop representation must be available!");
    auto& Defs = LDLILoop.first;
    Actual = buildPragma(Actual, Defs);
    Actual += '\n';
    Rewriter.InsertTextBefore(ForStmt->getLocStart(), Actual);
    auto& Lives = LDLILoop.second;
    GetActual = buildPragma(GetActual, Lives);
    Rewriter.InsertTextAfterToken(ForStmt->getLocEnd(), GetActual);
  }
}

SmallString<128>& ClangDVMParalleization::buildPragma(SmallString<128> Prefix,
  llvm::DenseSet<DIVariable*> Vars) {
  auto& pragma = Prefix;
  bool flag = false;
  for (auto var : Vars) {
    if (var == nullptr)
      continue;
    if (flag)
      pragma += ",";
    pragma += var->getName();
    flag = true;
  }
  pragma += ')';
  return pragma;
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

    auto& SrcMgr = mTfmCtx->getRewriter().getSourceMgr();
    auto& Rewriter = mTfmCtx->getRewriter();
    auto &Provider = getAnalysis<ClangDVMParalleizationProvider>(*F);    
    findLoopDefLiveInfo(F, Provider, Rewriter);
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
INITIALIZE_PASS_DEPENDENCY(DFRegionInfoPass)
INITIALIZE_PASS_DEPENDENCY(DIEstimateMemoryPass)
INITIALIZE_PASS_DEPENDENCY(DIDependencyAnalysisPass)
INITIALIZE_PASS_DEPENDENCY(ClangDVMParalleizationProvider)
INITIALIZE_PASS_DEPENDENCY(TransformationEnginePass)
INITIALIZE_PASS_DEPENDENCY(MemoryMatcherImmutableWrapper)
INITIALIZE_PASS_DEPENDENCY(GlobalOptionsImmutableWrapper)
INITIALIZE_PASS_DEPENDENCY(DIMemoryEnvironmentWrapper)
INITIALIZE_PASS_DEPENDENCY(DIMemoryTraitPoolWrapper)
INITIALIZE_PASS_DEPENDENCY(ClangDVMServerProvider)
INITIALIZE_PASS_DEPENDENCY(ClangDVMServerResponse)
INITIALIZE_PASS_DEPENDENCY(ParallelLoopPass)
INITIALIZE_PASS_DEPENDENCY(CanonicalLoopPass)
INITIALIZE_PASS_DEPENDENCY(ClangRegionCollector)
INITIALIZE_PASS_DEPENDENCY(LoopDefLiveVarInfoPass)
INITIALIZE_PASS_IN_GROUP_END(ClangDVMParalleization, "clang-dvm-parallel",
                             "DVM Based Parallelization (Clang)", false,
                             false,
                             TransformationQueryManager::getPassRegistry())
