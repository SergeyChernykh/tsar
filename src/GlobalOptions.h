//===--- GlobalOptions.h - Global Command Line Options ----------*- C++ -*-===//
//
//                       Traits Static Analyzer (SAPFOR)
//
//===----------------------------------------------------------------------===//
//
// This file declares classes to store command line options that should be
// accessed from different places. For example, some options are necessary for
// multiple passes. A special ImmutableWrapperPass can be used to access such
// options.
//
//===----------------------------------------------------------------------===//

#ifndef TSAR_GLOBAL_OPTIONS_H
#define TSAR_GLOBAL_OPTIONS_H

#include "tsar_pass.h"
#include <utility.h>
#include <llvm/Pass.h>

namespace tsar {
struct GlobalOptions {
  /// Print only names of files instead of full paths.
  bool PrintFilenameOnly = false;
};
}

namespace llvm {
/// Proposes access to global options.
class GlobalOptionsImmutableWrapper :
  public ImmutablePass, private bcl::Uncopyable {

public:
  /// Pass identification, replacement for typeid.
  static char ID;

  /// Creates pass with unspecified options.
  GlobalOptionsImmutableWrapper() : ImmutablePass(ID) {
    initializeGlobalOptionsImmutableWrapperPass(
      *PassRegistry::getPassRegistry());
  }

  /// Creates pass and associates it with a specified list of options.
  explicit GlobalOptionsImmutableWrapper(const tsar::GlobalOptions *Options) :
      ImmutablePass(ID), mOptions(Options) {
    assert(Options && "List of options must not be null!");
    initializeGlobalOptionsImmutableWrapperPass(
      *PassRegistry::getPassRegistry());
  }

  /// Returns a list of global options. List of options must be specified.
  const tsar::GlobalOptions & getOptions() const noexcept {
    assert(isSpecified() && "List of global options must be specified!");
    return *mOptions;
  }

  /// Returns true if a list of options is specified.
  bool isSpecified() const noexcept { return mOptions; }

private:
  const tsar::GlobalOptions *mOptions = nullptr;
};
}

#endif//TSAR_GLOBAL_OPTIONS_H