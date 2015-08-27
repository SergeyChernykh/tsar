//===------- tsar_df_loop.h - Represent a data-flow graph ------*- C++ -*-===//
//
//                       Traits Static Analyzer (SAPFOR)
//
//===--------------------------------------------------------------------===//
//
// This file defines functions and classes to represent a data-flow graph.
// The graph could be used in a data-flow framework to solve data-flow problem.
// In some cases it is convinient to use hierarchy of nodes. Some nodes are
// treated as regions which contain other nodes. LLVM-style RTTI for hierarch
// of classes that represented different nodes is avaliable.
//
// There are following main elements in this file:
// * Classes which is used to represent nodes and regions in a data-flow graph.
// * Functions, to build hierarchy of regions.
//===--------------------------------------------------------------------===//

#ifndef TSAR_LOOP_BODY_H
#define TSAR_LOOP_BODY_H

#include <llvm/ADT/GraphTraits.h>
#include "llvm/Support/Casting.h"
#include <vector>
#include <utility.h>
#include "tsar_data_flow.h"
#include "tsar_graph.h"
#include "declaration.h"

namespace llvm {
class Loop;
class BasicBlock;
}

namespace tsar {
/// \brief Representation of a node in a data-flow framework.
///
/// The following kinds of nodes are supported: basic block, body of the
/// natural loop, entry point of the graph which will be analyzed.
/// LLVM-style RTTI for hierarch of classes that represented different nodes
/// is avaliable.
/// \par In some cases it is convinient to use hierarchy of nodes. Some nodes
/// are treated as regions which contain other nodes. Such regions we call
/// parent nodes.
class DFNode : public tsar::SmallDFNode<DFNode, 8> {
public:
  /// Kind of a node.
  /// If you add a new kind of region it should be in the range between
  /// FIRST_KIND_REGION and LAST_KIND_REGION
  enum Kind {
    FIRST_KIND = 0,
    KIND_BLOCK = FIRST_KIND,
    KIND_ENTRY,

    FIRST_KIND_REGION,
    KIND_LOOP = FIRST_KIND_REGION,
    LAST_KIND_REGION = KIND_LOOP,

    LAST_KIND = KIND_ENTRY,
    INVALID_KIND,
    NUMBER_KIND = INVALID_KIND,
  };

  /// Creates a new node of the specified type.
  explicit DFNode(Kind K) : mKind(K), mParent(nullptr) {}

  /// Desctructor.
  virtual ~DFNode() {
#ifdef DEBUG
    mKind = INVALID_KIND;
    mParent = nullptr;
#endif
  }

  /// Returns the kind of the region.
  Kind getKind() const { return mKind; }

  /// Returns a parent node.
  DFNode * getParent() { return mParent; }

  /// Returns a parent node.
  const DFNode * getParent() const { return mParent; }

private:
  friend class DFRegion;
  Kind mKind;
  DFNode *mParent;
};

/// Representation of an entry node in a data-flow framework.
class DFEntry : public DFNode {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const DFNode *R) {
    return R->getKind() == KIND_ENTRY;
  }

  /// \brief Ctreates representation of the entry node.
  DFEntry() : DFNode(KIND_ENTRY) {}
};

/// \brief Representation of a region in a data-flow framework.
///
/// In some cases it is convinient to use hierarchy of nodes. Some nodes
/// are treated as regions which contain other nodes.
/// LLVM-style RTTI for hierarch of classes that represented different regions
/// is avaliable.
class DFRegion : public DFNode {
public:

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const DFNode *N) {
    return FIRST_KIND_REGION <= N->getKind() && 
      N->getKind() <= LAST_KIND_REGION;
  }

  /// Creates a new node of the specified type.
  explicit DFRegion(Kind K) :DFNode(K), mEntry(nullptr) {}

  /// \brief Deletes all nodes in the region.
  ///
  /// A memory which was allocated for the nodes is freed.
  ~DFRegion() {
    for (DFNode *N : mNodes)
      delete N;
    delete mEntry;
  }

  /// This type used to iterate over all nodes in the region body.
  typedef std::vector<DFNode *>::const_iterator nodes_iterator;

  /// This type used to iterate over internal regions.
  typedef std::vector<DFRegion *>::const_iterator regions_iterator;

  /// Get the number of nodes in this region.
  unsigned getNumNodes() const { return mNodes.size(); }

  /// Get a list of the nodes which make up this region body.
  const std::vector<DFNode *> & getNodes() const { return mNodes; }

  /// Returns iterator that points to the beginning of the nodes list.
  nodes_iterator nodes_begin() const { return mNodes.begin(); }

  /// Returns iterator that points to the ending of the nodes list.
  nodes_iterator nodes_end() const { return mNodes.end(); }

  /// Get the number of internal regions.
  unsigned getNumRegions() const { return mRegions.size(); }

  // Get a list of internal regions.
  const std::vector<DFRegion *> & getRegions() const { return mRegions; }

  /// Returns iterator that points to the beginning of the internal regions.
  regions_iterator regions_begin() const { return mRegions.begin(); }

  /// Returns iterator that points to the ending of the internal regions.
  regions_iterator regions_end() const { return mRegions.end(); }

  /// \brief Returns the entry-point of the data-flow graph.
  ///
  /// The result of this method is an entry point which is necessary to solve
  /// a data-flow problem. A node which is treated as entry depends on a region
  /// and it might not be essential in an original data-flow graph.
  /// For example, in case of loop, the entry node is not a header of the loop,
  /// this node is a predecessor of the header.
  /// \attention This node should not contained in  list of nodes which is
  /// a result of the getNodes() method. 
  DFNode * getEntryNode() const {
    assert(mEntry && "There is no entry node in the graph!");
    return mEntry;
  }

  /// \brief Inserts a new node at the end of the list of nodes.
  ///
  /// \attention The inserted node falls under the control of the region and
  /// will be destroyed at the same time when the region will be destroyed.
  /// \pre
  /// - A new node can not take a null value.
  /// - The node should be differ from other nodes of the graph.
  void addNode(DFNode *N) {
    assert(N && "Node must not be null!");
    assert(KIND_FIRST <= N->getKind() && N->getKind() <= KIND_LAST &&
      "Unknown kind of a node!");
    assert(N != mEntry && "Only one entry node must be in the region!");
#ifdef DEBUG
    for (DFNode *Node : mNodes)
      assert(N != Node &&
        "The node must not be contained in the region!");
#endif
    N->mParent = this;
    if (llvm::isa<DFEntry>(N)) {
      mEntry = N;
      return;
    }
    mNodes.push_back(N);
    if (DFRegion *R = llvm::dyn_cast<DFRegion>(N))
      mRegions.push_back(R);
  }
private:
  std::vector<DFNode *> mNodes;
  std::vector<DFRegion *> mRegions;
  DFNode *mEntry;
};

/// \brief Representation of a loop in a data-flow framework.
///
/// Instance of this class is used to represent abstraction of a loop
/// in data-flow framework. This class should be used only to solve
/// data-flow problem. The loop can be collapsed to one abstract node
/// to simplify the data-flow graph that contains this loop. If this loop
/// has inner loops they also can be collapsed. So the abstraction of this loop
/// can internally contain nodes of following types: basic block,
/// collapsed inner loop and entry node.
class DFLoop : public DFRegion {
public:
  /// This type used to iterate over all exiting nodes in the loop body.
  typedef llvm::SmallPtrSet<DFNode *, 8>::const_iterator exiting_iterator;

  /// This type used to iterate over all latch nodes in the loop body.
  typedef llvm::SmallPtrSet<DFNode *, 8>::const_iterator latch_iterator;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const DFNode *N) {
    return N->getKind() == KIND_LOOP;
  }

  /// \brief Creates representation of the loop.
  ///
  /// \pre The loop argument can not take a null value.
  explicit DFLoop(llvm::Loop *L) : DFRegion(KIND_LOOP), mLoop(L) {
    assert(L && "Loop must not be null!");
  }

  /// Get the loop.
  llvm::Loop * getLoop() const { return mLoop; }

  /// \brief Specifies an exiting node of the data-flow graph.
  ///
  /// Multiple nodes can be specified.
  void setExitingNode(DFNode *N) {
    assert(N && "Node must not be null!");
    mExitingNodes.insert(N);
  }

  /// Get a list of the exiting nodes of this loop.
  const llvm::SmallPtrSet<DFNode *, 8> & getExitingNodes() const { 
    return mExitingNodes;
  }

  /// Returns iterator that points to the beginning of the exiting nodes list.
  exiting_iterator exiting_begin() const { return mExitingNodes.begin(); }

  /// Returns iterator that points to the ending of the exiting nodes list.
  exiting_iterator exiting_end() const { return mExitingNodes.end(); }

  ///\brief  Returns true if the node is an exiting node of this loop.
  ///
  /// Exiting node is a node which is inside of the loop and 
  /// have successors outside of the loop.
  bool isLoopExiting(const DFNode *N) const {
    return mExitingNodes.count(const_cast<DFNode *>(N));
  }

  /// \brief Specifies an latch node of the data-flow graph.
  ///
  /// Multiple nodes can be specified.
  void setLatchNode(DFNode *N) {
    assert(N && "Node must not be null!");
    mLatchNodes.insert(N);
  }

  /// Get a list of the latch nodes of this loop.
  const llvm::SmallPtrSet<DFNode *, 8> & getLatchNodes() const {
    return mLatchNodes;
  }

  /// Returns iterator that points to the beginning of the latch nodes list.
  latch_iterator latch_begin() const { return mLatchNodes.begin(); }

  /// Returns iterator that points to the ending of the latch nodes list.
  latch_iterator latch_end() const { return mLatchNodes.end(); }

  ///\brief  Returns true if the node is an latch node of this loop.
  ///
  /// A latch node is a node that contains a branch back to the header.
  bool isLoopLatch(const DFNode *N) const {
    return mLatchNodes.count(const_cast<DFNode *>(N));
  }

private:
  llvm::SmallPtrSet<DFNode *, 8> mExitingNodes;
  llvm::SmallPtrSet<DFNode *, 8> mLatchNodes;
  llvm::Loop *mLoop;
};


/// \brief Representation of a basic block in a data-flow framework.
///
/// Instance of this class is used to represent abstraction of a basic block
/// in data-flow framework. This class should be used only to solve
/// data-flow problem.
class DFBlock : public DFNode {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const DFNode *R) {
    return R->getKind() == KIND_BLOCK;
  }

  /// \brief Ctreates representation of the block.
  ///
  /// \pre The block argument can not take a null value.
  explicit DFBlock(llvm::BasicBlock *B) : DFNode(KIND_BLOCK), mBlock(B) {
    assert(B && "Block must not be null!");
  }

  /// Get the block.
  llvm::BasicBlock * getBlock() const { return mBlock; }

private:
  llvm::BasicBlock *mBlock;
};

/// \brief Builds hierarchy of regions for the specified loop nest.
///
/// This function treats a loop nest as hierarchy of regions. Each region is
/// an abstraction of an inner loop. Only natural loops will be treated as a
/// region other loops will be ignored.
/// \param [in, out] L An outermost loop in the nest, it can not be null.
DFLoop * buildLoopRegion(llvm::Loop *L);
}
#endif//TSAR_LOOP_BODY_H