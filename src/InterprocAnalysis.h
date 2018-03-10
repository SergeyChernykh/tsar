#include "tsar_pass.h"
#include <bcl/utility.h>
#include <llvm/Pass.h>
#include <llvm/IR/Function.h>
#include <llvm/Analysis/CallGraphSCCPass.h>
#include <clang/Basic/SourceLocation.h>
#include <vector>

namespace {
struct FuncCallee {
  llvm::Function *Callee;
  std::vector<clang::SourceLocation> Locations;
  FuncCallee(llvm::Function *F) : Callee(F) {}
};
}

namespace tsar {
typedef llvm::DenseMap<llvm::Function *,
    std::vector<FuncCallee>> InterprocAnalysisInfo;
}

namespace llvm {
class InterprocAnalysisPass :
    public CallGraphSCCPass, private bcl::Uncopyable {
public:
  static char ID;
  InterprocAnalysisPass() : CallGraphSCCPass(ID) {
    initializeInterprocAnalysisPassPass(*PassRegistry::getPassRegistry());
  }
  tsar::InterprocAnalysisInfo & getInterprocAnalysisInfo() noexcept {
    return mInterprocAnalysisInfo;
  }
  const tsar::InterprocAnalysisInfo &
      getInterprocAnalysisInfo() const noexcept {
    return mInterprocAnalysisInfo;
  }
  bool runOnSCC(CallGraphSCC &SCC) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
private:
  tsar::InterprocAnalysisInfo mInterprocAnalysisInfo;
};
}
