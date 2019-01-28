//===- DIGlobalRetriever.cpp - Global Debug Info Retriever ------*- C++ -*-===//
//
//                       Traits Static Analyzer (SAPFOR)
//
// Copyright 2018 DVM System Group
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
// This file implements a pass which retrieves some debug information for
// global values if it is not presented in LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "tsar_pass.h"
#include "KnownFunctionTraits.h"
#include "tsar_transformation.h"
#include "tsar_utility.h"
#include <bcl/utility.h>
#include <clang/Basic/LangOptions.h>
#include <clang/AST/Decl.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include "llvm/Support/Path.h"

using namespace clang;
using namespace llvm;
using namespace tsar;

namespace {
/// This retrieves some debug information for global values if it is not
/// presented in LLVM IR ('sapfor.dbg' metadata will be attached to globals).
class DIGlobalRetrieverPass : public ModulePass, private bcl::Uncopyable {
public:
  static char ID;

  DIGlobalRetrieverPass() : ModulePass(ID) {
    initializeDIGlobalRetrieverPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.getPreservesAll();
  }
};
} // namespace

char DIGlobalRetrieverPass::ID = 0;
INITIALIZE_PASS(DIGlobalRetrieverPass, "global-diretriever",
                "Global Debug Info Retriever", true, false)

bool DIGlobalRetrieverPass::runOnModule(llvm::Module &M) {
  auto *TEP = getAnalysisIfAvailable<TransformationEnginePass>();
  if (!TEP)
    return false;
  auto *TfmCtx = TEP->getContext(M);
  if (!TfmCtx || !TfmCtx->hasInstance())
    return false;
  auto &Ctx = M.getContext();
  auto &SrcMgr = TfmCtx->getRewriter().getSourceMgr();
  auto &LangOpts = TfmCtx->getRewriter().getLangOpts();
  auto CUItr = M.debug_compile_units_begin();
  assert(CUItr != M.debug_compile_units_end() &&
         "At least one compile unit must be available!");
  auto CU =
    std::distance(CUItr, M.debug_compile_units_end()) == 1 ? *CUItr : nullptr;
  SmallString<256> CWD;
  auto DirName = CU ? CUItr->getDirectory()
                    : (llvm::sys::fs::current_path(CWD), StringRef(CWD));
  auto *FileCU = CU ? CU->getFile() : nullptr;
  DIBuilder DIB(M);
  for (auto &GlobalVar : M.globals()) {
    if (auto *MD = findMetadata(&GlobalVar))
      continue;
    DIFile *File = FileCU;
    unsigned Line = 0;
    // A name should be specified for global variables, otherwise LLVM IR is
    // considered corrupted.
    StringRef Name = "sapfor.var";
    if (auto D = TfmCtx->getDeclForMangledName(GlobalVar.getName())) {
      auto FName = SrcMgr.getFilename(SrcMgr.getExpansionLoc(D->getLocStart()));
      File = DIB.createFile(FName, DirName);
      Line = SrcMgr.getPresumedLineNumber(
        SrcMgr.getExpansionLoc(D->getLocStart()));
      if (auto ND = dyn_cast<NamedDecl>(D))
        Name = ND->getName();
    }
    /// TODO (kaniandr@gmail.com): we create a stub instead of an appropriate
    /// type because type must not be set to nullptr. We mark such type as
    /// artificial type with name "sapfor.type", however may be this is not
    /// a good way to distinguish such types?
    auto DIBasicTy = DIB.createBasicType(
      "char", llvm::Type::getInt1Ty(Ctx)->getScalarSizeInBits(),
      dwarf::DW_ATE_unsigned_char);
    auto PtrSize = M.getDataLayout().getPointerSizeInBits(
      GlobalVar.getType()->getAddressSpace());
    auto DITy = DIB.createArtificialType(
      DIB.createPointerType(DIBasicTy, PtrSize, 0, None, "sapfor.type"));
    auto *GV = DIGlobalVariable::getDistinct(
      Ctx, File, Name, GlobalVar.getName(), File, Line, DITy,
      GlobalVar.hasLocalLinkage(), GlobalVar.isDeclaration(), nullptr, 0);
    auto *GVE =
      DIGlobalVariableExpression::get(Ctx, GV, DIExpression::get(Ctx, {}));
    GlobalVar.setMetadata("sapfor.dbg", GVE);
  }
  for (auto &F : M.functions()) {
    if (F.getSubprogram())
      continue;
    if (F.isIntrinsic() && (isDbgInfoIntrinsic(F.getIntrinsicID()) ||
        isMemoryMarkerIntrinsic(F.getIntrinsicID())))
      continue;
    DIFile *File = FileCU;
    unsigned Line = 0;
    auto Flags = DINode::FlagZero;
    MDString *Name = nullptr;
    if (auto *D = TfmCtx->getDeclForMangledName(F.getName())) {
      auto FName = SrcMgr.getFilename(SrcMgr.getExpansionLoc(D->getLocStart()));
      File = DIB.createFile(FName, DirName);
      Line = SrcMgr.getPresumedLineNumber(
        SrcMgr.getExpansionLoc(D->getLocStart()));
      if (auto *FD = dyn_cast<FunctionDecl>(D)) {
        Name = MDString::get(Ctx, FD->getName());
        if (FD->hasPrototype())
          Flags |= DINode::FlagPrototyped;
        if (FD->isImplicit())
          Flags |= DINode::FlagArtificial;
      }
    }
    auto *SP = DISubprogram::getDistinct(Ctx, File, Name,
      MDString::get(Ctx, F.getName()), File, Line, nullptr, F.hasLocalLinkage(),
      !F.isDeclaration(), Line, nullptr, 0, 0, 0, Flags, LangOpts.Optimize,
      !F.isDeclaration() ? CU : nullptr);
    F.setMetadata("sapfor.dbg", SP);
  }
  return true;
}

ModulePass *llvm::createDIGlobalRetrieverPass() {
  return new DIGlobalRetrieverPass();
}
