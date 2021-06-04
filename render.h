#pragma once

#include <nanovdb/NanoVDB.h>
#include <string>

enum class RenderMode { DeltaTracking, RatioTracking, MIS, SpectralMIS };

inline RenderMode render_mode_from_str(const std::string &str) {
#define RENDER_MODE_CASE(name)                                                 \
  do {                                                                         \
    if (str == #name) {                                                        \
      return RenderMode::name;                                                 \
    }                                                                          \
  } while (false)

  RENDER_MODE_CASE(DeltaTracking);
  RENDER_MODE_CASE(RatioTracking);
  RENDER_MODE_CASE(MIS);
  RENDER_MODE_CASE(SpectralMIS);

#undef RENDER_MODE_CASE
  return RenderMode::DeltaTracking;
}

struct Scene {
  int frame_width;
  int frame_height;
  const nanovdb::FloatGrid *volume_grid;
  float camera_pos[3];
  float light_dir[3];
  float light_color[3];
  float light_cos_angle;
  int spp;
  float max_value;
  struct PhaseFunction {
    float color[3];
    float cos_angle;
    float diffuse;
  } phase_func;
  RenderMode mode;
};

void render(const Scene &scene, float *d_image);