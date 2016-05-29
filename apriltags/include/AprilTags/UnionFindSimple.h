#ifndef UNIONFINDSIMPLE_H
#define UNIONFINDSIMPLE_H

#include <vector>

namespace AprilTags {

//! Implementation of disjoint set data structure using the union-find algorithm
class UnionFindSimple {

public:
  explicit UnionFindSimple(int maxId) : data(maxId) {
    init();
  };

  //! Identifies parent ids and sizes.
  struct Data {
    int id;
    int size;
  };
  
  int getSetSize(int thisId) { return data[getRepresentative(thisId)].size; }

  int getRepresentative(int thisId);

  //! Returns the id of the merged node.
  /*  @param aId
   *  @param bId
   */
  int connectNodes(int aId, int bId);

  std::vector<Data> getData() { return data; }

  void printDataVector() const;

private:
  void init();
  
  std::vector<Data> data;
};

} // namespace

#endif
