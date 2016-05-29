#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "AprilTags/Edge.h"
#include "AprilTags/FloatImage.h"
#include "AprilTags/MathUtil.h"
#include "AprilTags/UnionFindSimple.h"

namespace AprilTags {

float const Edge::minMag = 0.004f;
float const Edge::maxEdgeCost = 30.f * float(M_PI) / 180.f;
int const Edge::WEIGHT_SCALE = 100;
float const Edge::thetaThresh = 100;
float const Edge::magThresh = 1200;

__global__ void mykernel(void) {
}

int Edge::example_cuda_function() {
  mykernel<<<1,1>>>();
	return 1;
}

int Edge::edgeCost(float  theta0, float theta1, float mag1) {
  if (mag1 < minMag)  // mag0 was checked by the main routine so no need to recheck here
    return -1;

  const float thetaErr = std::abs(MathUtil::mod2pi(theta1 - theta0));
  if (thetaErr > maxEdgeCost)
    return -1;

  const float normErr = thetaErr / maxEdgeCost;
  return (int) (normErr*WEIGHT_SCALE);
}


void Edge::calcEdges(float theta0, int x, int y,
		     const FloatImage& theta, const FloatImage& mag,
         std::vector<Edge> &edges, size_t &nEdges) {

  int width = theta.getWidth();
  int thisPixel = y*width+x;

  // horizontal edge
  int cost1 = edgeCost(theta0, theta.get(x+1,y), mag.get(x+1,y));
  if (cost1 >= 0) {
    edges[nEdges].cost = cost1;
    edges[nEdges].pixelIdxA = thisPixel;
    edges[nEdges].pixelIdxB = y*width+x+1;
    ++nEdges;
  }

  // vertical edge
  int cost2 = edgeCost(theta0, theta.get(x, y+1), mag.get(x,y+1));
  if (cost2 >= 0) {
    edges[nEdges].cost = cost2;
    edges[nEdges].pixelIdxA = thisPixel;
    edges[nEdges].pixelIdxB = (y+1)*width+x;
    ++nEdges;
  }
  
  // downward diagonal edge
  int cost3 = edgeCost(theta0, theta.get(x+1, y+1), mag.get(x+1,y+1));
  if (cost3 >= 0) {
    edges[nEdges].cost = cost3;
    edges[nEdges].pixelIdxA = thisPixel;
    edges[nEdges].pixelIdxB = (y+1)*width+x+1;
    ++nEdges;
  }

  // updward diagonal edge
  int cost4 = (x == 0) ? -1 : edgeCost(theta0, theta.get(x-1, y+1), mag.get(x-1,y+1));
  if (cost4 >= 0) {
    edges[nEdges].cost = cost4;
    edges[nEdges].pixelIdxA = thisPixel;
    edges[nEdges].pixelIdxB = (y+1)*width+x-1;
    ++nEdges;
  }
}

__device__ int uf_getRep(UnionFindSimple::Data* data, int thisId) {
  if (data[thisId].id == thisId)
    return thisId;
  // otherwise, recurse...
  int root = uf_getRep(data, data[thisId].id);
  // short circuit the path
  data[thisId].id = root;
  return root;
}

__device__ int uf_connectNodes(UnionFindSimple::Data* data, int aId, int bId) {
  int aRoot = uf_getRep(data, aId);
  int bRoot = uf_getRep(data, bId);

  if (aRoot == bRoot)
    return aRoot;

  int asz = data[aRoot].size;
  int bsz = data[bRoot].size;

  if (asz > bsz) {
    data[bRoot].id = aRoot;
    data[aRoot].size += bsz;
    return aRoot;
  } else {
    data[aRoot].id = bRoot;
    data[bRoot].size += asz;
    return bRoot;
  }
}

__device__ float mod2pi(float vin) {
  const float twopi = 2 * (float)M_PI;
  const float twopi_inv = 1.f / (2.f * (float)M_PI);
  float absv = std::abs(vin);
  float q = absv*twopi_inv + 0.5f;
  int qi = (int) q;
  float r = absv - qi*twopi;
  return (vin<0) ? -r : r;
}

__device__ float mod2pi(float ref, float v) { 
  return ref + mod2pi(v-ref); 
}

__device__ int uf_getSetSize(UnionFindSimple::Data* data, int thisId) {
  // Check whether it needs shallow copy or deep copy
  return data[uf_getRep(data, thisId)].size;
}

__device__ float min(float a, float b) {
  return a > b ? b : a;
}

__device__ float max(float a, float b) {
  return a < b ? a : b;
}

__global__ void mergeEdges_kernel(Edge* edges, UnionFindSimple::Data* data
                ,float* tmin, float* tmax, float* mmin, float *mmax) {

  int ida = uf_getRep(data, edges[blockIdx.x].pixelIdxA);
  int idb = uf_getRep(data, edges[blockIdx.x].pixelIdxB);

  if(ida != idb) {
    int sza = uf_getSetSize(data, ida);
    int szb = uf_getSetSize(data, idb);
    float tmina = tmin[ida], tmaxa = tmax[ida];
    float tminb = tmin[idb], tmaxb = tmax[idb];

    float costa = (tmaxa-tmina);
    float costb = (tmaxb-tminb);

    // bshift will be a multiple of 2pi that aligns the spans of 'b' with 'a'
    // so that we can properly take the union of them.
    float bshift = mod2pi((tmina+tmaxa)/2, (tminb+tmaxb)/2) - (tminb+tmaxb)/2;

    float tminab = min(tmina, tminb + bshift);
    float tmaxab = max(tmaxa, tmaxb + bshift);

    if (tmaxab-tminab > 2*(float)M_PI) // corner case that's probably not too useful to handle correctly, oh well.
      tmaxab = tminab + 2*(float)M_PI;

    float mminab = min(mmin[ida], mmin[idb]);
    float mmaxab = max(mmax[ida], mmax[idb]);

    // merge these two clusters?
    float costab = (tmaxab - tminab);
    if (costab <= (min(costa, costb) + Edge::thetaThresh/(sza+szb)) &&
	(mmaxab-mminab) <= min(mmax[ida]-mmin[ida], mmax[idb]-mmin[idb]) + Edge::magThresh/(sza+szb)) {
	
      int idab = uf_connectNodes(data, ida, idb);
	
      tmin[idab] = tminab;
      tmax[idab] = tmaxab;
	
      mmin[idab] = mminab;
      mmax[idab] = mmaxab;
    }
  }
}

void Edge::mergeEdges_cuda(std::vector<Edge> &edges, UnionFindSimple &uf,
		      float tmin[], float tmax[], float mmin[], float mmax[]) {
  
  float *d_tmin, *d_tmax, *d_mmin, *d_mmax;
  int edge_size = sizeof(edges);
  int data_size = uf.getData().size();
  std::vector<UnionFindSimple::Data> data = uf.getData();
  Edge d_edges[edge_size];
  UnionFindSimple::Data d_data[data_size];
  std::copy(edges.begin(), edges.end(), d_edges);
  std::copy(data.begin(), data.end(), d_data);
  
  // Allocate space for device copies
  
  cudaMalloc((void **)&d_tmin, sizeof(tmin));
  cudaMalloc((void **)&d_tmax, sizeof(tmax));
  cudaMalloc((void **)&d_mmin, sizeof(mmin));
  cudaMalloc((void **)&d_mmax, sizeof(mmax));

  cudaMemcpy(d_tmin, tmin, sizeof(tmin), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tmax, tmax, sizeof(tmax), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mmin, mmin, sizeof(mmin), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mmax, mmax, sizeof(mmax), cudaMemcpyHostToDevice);

  mergeEdges_kernel<<<edge_size, 1>>>(d_edges, d_data, d_tmin, d_tmax, d_mmin, d_mmax);

  cudaMemcpy(d_tmin, tmin, sizeof(tmin), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_tmax, tmax, sizeof(tmax), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_mmin, mmin, sizeof(mmin), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_mmax, mmax, sizeof(mmax), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_tmin);
  cudaFree(d_tmax);
  cudaFree(d_mmin);
  cudaFree(d_mmax);

}

void Edge::mergeEdges(std::vector<Edge> &edges, UnionFindSimple &uf,
		      float tmin[], float tmax[], float mmin[], float mmax[]) {

  for (size_t i = 0; i < edges.size(); i++) {
    int ida = edges[i].pixelIdxA;
    int idb = edges[i].pixelIdxB;

    ida = uf.getRepresentative(ida);
    idb = uf.getRepresentative(idb);
      
    if (ida == idb)
      continue;

    int sza = uf.getSetSize(ida);
    int szb = uf.getSetSize(idb);

    float tmina = tmin[ida], tmaxa = tmax[ida];
    float tminb = tmin[idb], tmaxb = tmax[idb];

    float costa = (tmaxa-tmina);
    float costb = (tmaxb-tminb);

    // bshift will be a multiple of 2pi that aligns the spans of 'b' with 'a'
    // so that we can properly take the union of them.
    float bshift = mod2pi((tmina+tmaxa)/2, (tminb+tmaxb)/2) - (tminb+tmaxb)/2;

    float tminab = min(tmina, tminb + bshift);
    float tmaxab = max(tmaxa, tmaxb + bshift);

    if (tmaxab-tminab > 2*(float)M_PI) // corner case that's probably not too useful to handle correctly, oh well.
      tmaxab = tminab + 2*(float)M_PI;

    float mminab = min(mmin[ida], mmin[idb]);
    float mmaxab = max(mmax[ida], mmax[idb]);

    // merge these two clusters?
    float costab = (tmaxab - tminab);
    if (costab <= (min(costa, costb) + Edge::thetaThresh/(sza+szb)) &&
	(mmaxab-mminab) <= min(mmax[ida]-mmin[ida], mmax[idb]-mmin[idb]) + Edge::magThresh/(sza+szb)) {
	
      int idab = uf.connectNodes(ida, idb);
	
      tmin[idab] = tminab;
      tmax[idab] = tmaxab;
	
      mmin[idab] = mminab;
      mmax[idab] = mmaxab;
    }
  }
}

} // namespace
