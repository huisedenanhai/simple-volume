#include "random.h"
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

__device__ __forceinline__ uint32_t reverse_bits_32(uint32_t n) {
  n = (n << 16) | (n >> 16);
  n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
  n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
  n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
  n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
  return n;
}

__device__ __forceinline__ float radical_inverse_base_2(uint32_t n) {
  return saturate(reverse_bits_32(n) * float(2.3283064365386963e-10));
}

// 0 <= i < n
__device__ __forceinline__ float2 hammersley_sample(uint32_t i, uint32_t n) {
  return make_float2((float)(i + 1) / (float)(n), radical_inverse_base_2(i));
}

__device__ __forceinline__ float
uniform_sample_cone(float u, float v, float cos_phi, float3 &d) {
  float theta = 2.0f * Pi * u;
  float y = 1.0f - u * (1.0f - cos_phi);
  auto r = sqrtf(max(0.0f, 1.0f - y * y));
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return 0.5f * InvPi / (1.0f - cos_phi);
}

// y goes upward
__device__ __forceinline__ float
uniform_sample_hemisphere(float u, float v, float3 &d) {
  return uniform_sample_cone(u, v, 0, d);
}

__device__ __forceinline__ float
uniform_sample_hemisphere(unsigned int &randState, float3 &d) {
  return uniform_sample_hemisphere(rnd(randState), rnd(randState), d);
}

__device__ __forceinline__ float
uniform_sample_sphere(float u, float v, float3 &d) {
  return uniform_sample_cone(u, v, -1.0f, d);
}

// y goes upward
__device__ __forceinline__ float
cosine_sample_hemisphere(float u, float v, float3 &d) {
  auto theta = 2.0f * Pi * u;
  auto r = sqrtf(v);
  auto y = sqrtf(max(1 - r * r, 0.0f));
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return y * InvPi;
}

__device__ __forceinline__ float
cosine_sample_hemisphere(unsigned int &randState, float3 &d) {
  return cosine_sample_hemisphere(rnd(randState), rnd(randState), d);
}

__device__ __forceinline__ float lerp(float a, float b, float t) {
  return a + t * (b - a);
}

__device__ __forceinline__ float4 lerp(float4 a, float4 b, float4 t) {
  return make_float4(lerp(a.x, b.x, t.x),
                     lerp(a.y, b.y, t.y),
                     lerp(a.z, b.z, t.z),
                     lerp(a.w, b.w, t.w));
}

__device__ __forceinline__ float4 lerp(float a, float b, float4 t) {
  return lerp(make_float4(a, a, a, a), make_float4(b, b, b, b), t);
}

__device__ __forceinline__ float inverse_lerp(float a, float b, float v) {
  return (v - a) / (b - a);
}

__device__ __forceinline__ float clamp(float v, float a, float b) {
  return max(min(v, b), a);
}

__device__ __forceinline__ float
sample_array(float *data, uint32_t count, float u) {
  uint32_t i = clamp(count * u, 0, count - 2);
  return lerp(data[i], data[i + 1], saturate(count * u - i));
}

__device__ __forceinline__ nanovdb::Ray<float> sample_camera_ray(
    const Scene &scene, int c, int r, float2 jitter = make_float2(0.0f, 0.0f)) {
  int frame_width = scene.frame_width;
  int frame_height = scene.frame_height;

  float2 uv = make_float2((float(c) + jitter.x) / float(frame_width),
                          (float(r) + jitter.y) / float(frame_height));
  float aspect = float(frame_width) / float(frame_height);

  auto &camera_pos = scene.camera_pos;
  nanovdb::Vec3<float> origin(camera_pos[0], camera_pos[1], camera_pos[2]);
  nanovdb::Vec3<float> direction(uv.x - 0.5f, (uv.y - 0.5f) / aspect, -1.0);
  direction.normalize();
  nanovdb::Ray<float> wRay(origin, direction);
  return wRay;
}

__global__ void render_kernel_raymarching(Scene scene, float3 *image) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  int frame_width = scene.frame_width;
  int frame_height = scene.frame_height;
  const int index = r * frame_width + c;

  if ((c >= frame_width) || (r >= frame_height))
    return;

  nanovdb::Vec3<float> light_direction(
      scene.light_dir[0], scene.light_dir[1], scene.light_dir[2]);
  light_direction.normalize();
  auto grid = scene.volume_grid;

  nanovdb::Ray<float> wRay = sample_camera_ray(scene, c, r);
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
  float3 contrib = make_float3(0.0f, 0.0f, 0.0f);
  float dt = 0.5f;
  float3 color = make_float3(scene.phase_func.color[0],
                             scene.phase_func.color[1],
                             scene.phase_func.color[2]);

  for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
    auto iPos = iRay(t);
    float sigma = acc.getValue(nanovdb::Coord::Floor(iPos));
    contrib += ray_march_transmittance(
                   grid, {grid->indexToWorldF(iPos), light_direction}, dt) *
               transmittance * dt * sigma * color * 0.25f * InvPi;
    transmittance *= 1.0f - sigma * dt;
  }
  image[index] = contrib;
}

__device__ __forceinline__ float3 get_light_emission(const Scene &scene,
                                                     const float3 &direction) {
  float3 light_dir =
      make_float3(scene.light_dir[0], scene.light_dir[1], scene.light_dir[2]);
  float attenuation =
      dot(light_dir, direction) >= scene.light_cos_angle ? 1.0f : 0.0f;
  auto &light_color = scene.light_color;
  return attenuation *
         make_float3(light_color[0], light_color[1], light_color[2]);
}

__device__ __forceinline__ float sample_free_flight(float u, float mu) {
  return -logf(1.0f - u) / mu;
}

__device__ __forceinline__ float3
vec_as_float3(const nanovdb::Vec3<float> &vec) {
  return make_float3(vec[0], vec[1], vec[2]);
}

__device__ __forceinline__ nanovdb::Vec3<float>
float3_as_vec(const float3 &vec) {
  return nanovdb::Vec3<float>(vec.x, vec.y, vec.z);
}

__global__ void render_kernel_delta_tracking(Scene scene, float3 *image) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  int frame_width = scene.frame_width;
  int frame_height = scene.frame_height;
  const int index = r * frame_width + c;

  if ((c >= frame_width) || (r >= frame_height))
    return;

  unsigned int seed = tea<4>(index, 11424);
  int spp = scene.spp;
  float aspect = float(frame_width) / float(frame_height);
  auto grid = scene.volume_grid;
  float max_value = scene.max_value;
  float3 color = make_float3(scene.phase_func.color[0],
                             scene.phase_func.color[1],
                             scene.phase_func.color[2]);
  float3 contrib = make_float3(0.0f, 0.0f, 0.0f);
  auto accessor = grid->tree().getAccessor();
  for (int i = 0; i < spp; i++) {
    float2 jitter = hammersley_sample(i, spp);
    nanovdb::Ray<float> w_ray = sample_camera_ray(scene, c, r, jitter);
    float3 factor = make_float3(1.0f, 1.0f, 1.0f);

    bool miss = false;
    for (int bounce = 0; !miss && bounce < 30; bounce++) {
      nanovdb::Ray<float> i_ray = w_ray.worldToIndexF(*grid);
      if (i_ray.clip(grid->tree().bbox()) == false) {
        miss = true;
        break;
      }

      float t = i_ray.t0();

      while (t < i_ray.t1()) {
        t += sample_free_flight(rnd(seed), max_value);
        auto i_pos = i_ray(t);
        float sigma = accessor.getValue(nanovdb::Coord::Floor(i_pos));
        if (rnd(seed) > sigma / max_value) {
          continue;
        } else {
          break;
        }
      }

      if (t >= i_ray.t1()) {
        miss = true;
        break;
      }

      // hit happens, sample a new direction
      auto next_origin = grid->indexToWorldF(i_ray(t));
      float3 next_dir;
      uniform_sample_sphere(rnd(seed), rnd(seed), next_dir);
      w_ray = nanovdb::Ray<float>(next_origin, float3_as_vec(next_dir));
      factor *= color;
    }

    if (miss) {
      contrib += factor * get_light_emission(scene, vec_as_float3(w_ray.dir()));
    }
  }
  image[index] = contrib / float(spp);
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
      printf("CUDA ERROR\n");                                                  \
    }                                                                          \
  } while (false)

void render(const Scene &scene, float *d_image) {
  printf("render\n");
  assert(scene.volume_grid);
  // auto kernel = render_kernel_raymarching;
  auto kernel = render_kernel_delta_tracking;
  launch2d(
      kernel, scene.frame_width, scene.frame_height, scene, (float3 *)d_image);
}