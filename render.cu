#include "random.h"
#include "render.h"
#include "vec_math.h"
#include <nanovdb/util/Ray.h>
#include <stdio.h>

__device__ __forceinline__ float3 array_as_float3(const float *vec) {
  return make_float3(vec[0], vec[1], vec[2]);
}

__device__ __forceinline__ float3
vec_as_float3(const nanovdb::Vec3<float> &vec) {
  return make_float3(vec[0], vec[1], vec[2]);
}

__device__ __forceinline__ nanovdb::Vec3<float>
float3_as_vec(const float3 &vec) {
  return nanovdb::Vec3<float>(vec.x, vec.y, vec.z);
}

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

__device__ __forceinline__ float uniform_sample_cone_pdf(float cos_phi) {
  return 0.5f * InvPi / (1.0f - cos_phi);
}

__device__ __forceinline__ float
uniform_sample_cone(float u, float v, float cos_phi, float3 &d) {
  float theta = 2.0f * Pi * u;
  float y = 1.0f - v * (1.0f - cos_phi);
  auto r = sqrtf(max(0.0f, 1.0f - y * y));
  d.x = r * cosf(theta);
  d.y = y;
  d.z = r * sinf(theta);
  return uniform_sample_cone_pdf(cos_phi);
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

struct TBN {
  float3 n;
  float3 t;
  float3 b;

  __device__ __forceinline__ float3 local_to_world(const float3 &v) const {
    return n * v.y + t * v.x + b * v.z;
  }
};

__device__ __forceinline__ TBN construct_tbn(const float3 &normal) {
  float3 n = normalize(normal);
  float3 up = make_float3(0.0f, 0.0f, 1.0f);
  if (n.z > 0.98f) {
    up = make_float3(1.0f, 0.0f, 0.0f);
  }
  float3 t = normalize(cross(up, n));
  float3 b = normalize(cross(t, n));
  TBN tbn{};
  tbn.n = n;
  tbn.t = t;
  tbn.b = b;
  return tbn;
}

__device__ __forceinline__ float3 get_light_emission(const Scene &scene,
                                                     const float3 &direction) {
  float3 light_dir = normalize(array_as_float3(scene.light_dir));
  float attenuation =
      dot(light_dir, direction) >= scene.light_cos_angle ? 1.0f : 0.0f;
  return attenuation * array_as_float3(scene.light_color);
}

__device__ __forceinline__ float sample_light(const Scene &scene,
                                              unsigned int &seed,
                                              float3 &direction,
                                              float3 &emission) {
  float3 local_dir;
  float pdf = uniform_sample_cone(
      rnd(seed), rnd(seed), scene.light_cos_angle, local_dir);
  float3 light_dir = normalize(array_as_float3(scene.light_dir));

  direction = construct_tbn(light_dir).local_to_world(local_dir);
  emission = array_as_float3(scene.light_color);
  return pdf;
}

__device__ __forceinline__ float sample_light_pdf(const Scene &scene,
                                                  float3 &direction) {
  float3 light_dir = normalize(array_as_float3(scene.light_dir));
  float attenuation =
      dot(light_dir, direction) >= scene.light_cos_angle ? 1.0f : 0.0f;
  return uniform_sample_cone_pdf(scene.light_cos_angle) * attenuation;
}

__device__ __forceinline__ float sample_free_flight(float u, float mu) {
  return -logf(1.0f - u) / mu;
}

__device__ __forceinline__ float eval_phase_function(float diffuse,
                                                     float cos_angle,
                                                     const float3 &curr_dir,
                                                     const float3 &next_dir) {
  float uniform_pdf = 0.25f * InvPi;
  float cone_pdf = 0.0f;
  if (dot(curr_dir, next_dir) > cos_angle) {
    cone_pdf = uniform_sample_cone_pdf(cos_angle);
  }

  return lerp(cone_pdf, uniform_pdf, clamp(diffuse, 0.0f, 1.0f));
}

__device__ __forceinline__ float sample_phase_function(unsigned int &seed,
                                                       float diffuse,
                                                       float cos_angle,
                                                       const float3 &curr_dir,
                                                       float3 &next_dir) {
  if (rnd(seed) < diffuse) {
    uniform_sample_sphere(rnd(seed), rnd(seed), next_dir);
  } else {
    float3 next_dir_local;
    uniform_sample_cone(rnd(seed), rnd(seed), cos_angle, next_dir_local);
    next_dir = construct_tbn(curr_dir).local_to_world(next_dir_local);
  }
  return eval_phase_function(diffuse, cos_angle, curr_dir, next_dir);
}

struct DeltaTrackResult {
  bool miss = false;
  float t = 0;
  float null_scatter_factor = 1.0f;
};

template <bool force_null_scatter = false>
__device__ __forceinline__ DeltaTrackResult
delta_track_ray(const nanovdb::FloatGrid *grid,
                const decltype(grid->getAccessor()) &accessor,
                unsigned int &seed,
                float max_sigma,
                float sigma_scale,
                nanovdb::Ray<float> &i_ray) {
  DeltaTrackResult res{};
  res.t = i_ray.t0();
  if (i_ray.clip(grid->tree().bbox()) == false) {
    res.miss = true;
    return res;
  }

  float &t = res.t;
  float &factor = res.null_scatter_factor;
  float last_null_scatter_factor = 1.0f;

  while (t < i_ray.t1()) {
    float scaled_max_value = max(max_sigma * sigma_scale, 1e-4);
    t += sample_free_flight(rnd(seed), scaled_max_value);
    factor *= last_null_scatter_factor;

    auto i_pos = i_ray(t);
    float sigma = accessor.getValue(nanovdb::Coord::Floor(i_pos)) * sigma_scale;
    float real_scatter_p = sigma / scaled_max_value;
    last_null_scatter_factor = 1.0f - real_scatter_p;

    if constexpr (force_null_scatter) {
      continue;
    }
    if (rnd(seed) > real_scatter_p) {
      continue;
    } else {
      break;
    }
  }

  if (t >= i_ray.t1()) {
    res.miss = true;
  }

  return res;
}

template <RenderMode mode>
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
  float sigma_scale = 1.0f;
  float3 color = make_float3(scene.phase_func.color[0],
                             scene.phase_func.color[1],
                             scene.phase_func.color[2]);
  float3 contrib = make_float3(0.0f, 0.0f, 0.0f);
  auto accessor = grid->tree().getAccessor();
  for (int i = 0; i < spp; i++) {
    float2 jitter = hammersley_sample(i, spp);
    nanovdb::Ray<float> w_ray = sample_camera_ray(scene, c, r, jitter);
    float3 factor = make_float3(1.0f, 1.0f, 1.0f);

    DeltaTrackResult hit{};
    float last_phase = 1.0f;
    int bounce = 0;
    for (; !hit.miss && bounce < 30; bounce++) {
      nanovdb::Ray<float> i_ray = w_ray.worldToIndexF(*grid);
      hit = delta_track_ray<false>(
          grid, accessor, seed, max_value, sigma_scale, i_ray);
      if (hit.miss) {
        break;
      }
      // hit happens, sample a new direction
      auto next_origin = grid->indexToWorldF(i_ray(hit.t));
      float3 curr_dir = normalize(vec_as_float3(w_ray.dir()));
      float3 next_dir;
      last_phase = sample_phase_function(seed,
                                         scene.phase_func.diffuse,
                                         scene.phase_func.cos_angle,
                                         curr_dir,
                                         next_dir);
      w_ray = nanovdb::Ray<float>(next_origin, float3_as_vec(next_dir));
      factor *= color;

      if constexpr (mode == RenderMode::RatioTracking ||
                    mode == RenderMode::MIS) {
        float3 light_dir;
        float3 light_emission;
        float pe = sample_light(scene, seed, light_dir, light_emission);
        nanovdb::Ray<float> w_light_ray(next_origin,
                                        float3_as_vec(normalize(light_dir)));
        nanovdb::Ray<float> i_light_ray = w_light_ray.worldToIndexF(*grid);
        auto rt = delta_track_ray<true>(
            grid, accessor, seed, max_value, sigma_scale, i_light_ray);
        float phase = eval_phase_function(scene.phase_func.diffuse,
                                          scene.phase_func.cos_angle,
                                          curr_dir,
                                          normalize(light_dir));
        float3 rt_contrib =
            factor * phase * rt.null_scatter_factor * light_emission / pe;
        if constexpr (mode == RenderMode::RatioTracking) {
          contrib += rt_contrib;
        }

        if constexpr (mode == RenderMode::MIS) {
          // balance heuristic
          float w_rt = pe / (pe + rt.null_scatter_factor * phase);
          contrib += rt_contrib * w_rt;
        }
      }
    }

    if (hit.miss) {
      if (mode != RenderMode::RatioTracking || bounce == 0) {
        // pure ratio tracking only handles one or more bounces
        float w_dt = 1.0f;
        if (mode == RenderMode::MIS && bounce != 0) {
          float pe =
              sample_light_pdf(scene, normalize(vec_as_float3(w_ray.dir())));
          w_dt = hit.null_scatter_factor * last_phase /
                 (pe + hit.null_scatter_factor * last_phase);
        }
        contrib += w_dt * factor *
                   get_light_emission(scene, vec_as_float3(w_ray.dir()));
      }
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

  auto launch = [&](auto kernel) {
    launch2d(kernel,
             scene.frame_width,
             scene.frame_height,
             scene,
             (float3 *)d_image);
  };

  switch (scene.mode) {
  case RenderMode::DeltaTracking:
    launch(render_kernel_delta_tracking<RenderMode::DeltaTracking>);
    break;
  case RenderMode::RatioTracking:
    launch(render_kernel_delta_tracking<RenderMode::RatioTracking>);
    break;
  case RenderMode::MIS:
    launch(render_kernel_delta_tracking<RenderMode::MIS>);
    break;
  case RenderMode::SpectralMIS:
    launch(render_kernel_delta_tracking<RenderMode::SpectralMIS>);
    break;
  }

  cudaDeviceSynchronize();
}