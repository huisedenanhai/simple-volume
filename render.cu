#include "render.h"
#include "vec_math.h"
#include <nanovdb/util/Ray.h>
#include <stdio.h>

__device__ float ray_march_transmittance(const nanovdb::FloatGrid *grid,
                                         const nanovdb::Ray<float> &wRay,
                                         float dt) {
  // transform the ray to the grid's index-space...
  nanovdb::Ray<float> iRay = wRay.worldToIndexF(*grid);
  // clip to bounds.
  if (iRay.clip(grid->tree().bbox()) == false) {
    return 1.0f;
  }
  // get an accessor.
  auto acc = grid->tree().getAccessor();
  // integrate along ray interval...
  float transmittance = 1.0f;
  for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
    float sigma = acc.getValue(nanovdb::Coord::Floor(iRay(t)));
    transmittance *= 1.0f - sigma * dt;
  }
  return transmittance;
}

__global__ void render_kernel(Scene scene, float3 *image) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  int frame_width = scene.frame_width;
  int frame_height = scene.frame_height;
  const int index = r * frame_width + c;

  if ((c >= frame_width) || (r >= frame_height))
    return;
  nanovdb::Vec3<float> light_direction(0, 1, 0);
  light_direction.normalize();
  if (r == 0 && c == 0) {
    printf("%f, %f, %f\n",
           light_direction[0],
           light_direction[1],
           light_direction[2]);
  }

  float2 uv = make_float2(float(c) / float(frame_width),
                          float(r) / float(frame_height));
  float aspect = float(frame_width) / float(frame_height);

  auto grid = scene.volume_grid;
  nanovdb::Vec3<float> origin(0, 20, 100);
  nanovdb::Vec3<float> direction(uv.x - 0.5f, (uv.y - 0.5f) / aspect, -1.0);
  direction.normalize();
  nanovdb::Ray<float> wRay(origin, direction);
  // transform the ray to the grid's index-space...
  nanovdb::Ray<float> iRay = wRay.worldToIndexF(*grid);
  // clip to bounds.
  if (iRay.clip(grid->tree().bbox()) == false) {
    image[index] = make_float3(0, 0, 0);
    return;
  }
  // get an accessor.
  auto acc = grid->tree().getAccessor();
  // integrate along ray interval...
  float transmittance = 1.0f;
  float contrib = 0.0f;
  float dt = 0.5f;
  for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
    auto iPos = iRay(t);
    float sigma = acc.getValue(nanovdb::Coord::Floor(iPos));
    contrib += ray_march_transmittance(
                   grid, {grid->indexToWorldF(iPos), light_direction}, dt) *
               transmittance * dt * sigma;
    transmittance *= 1.0f - sigma * dt;
  }
  image[index] = make_float3(1, 1, 1) * contrib;
}

template <typename Kernel, typename... Args>
void launch2d(Kernel &&k, int width, int height, Args &&... args) {
  dim3 block_size(1, 1);
  int grid_x = (width + block_size.x - 1) / block_size.x;
  int grid_y = (height + block_size.y - 1) / block_size.y;
  dim3 grid_size(grid_x, grid_y);
  printf("width %d, height %d\n", width, height);
  k<<<grid_size, block_size>>>(args...);
}

#define CHECK_CUDA_ERROR                                                       \
  do {                                                                         \
    cudaError_t e;                                                             \
    e = cudaGetLastError();                                                    \
    if (e != cudaSuccess) {                                                    \
      printf("fuck\n");                                                        \
    }                                                                          \
  } while (false)

void render(const Scene &scene, float *d_image) {
  printf("render\n");
  assert(scene.volume_grid);
  launch2d(render_kernel,
           scene.frame_width,
           scene.frame_height,
           scene,
           (float3 *)d_image);
}