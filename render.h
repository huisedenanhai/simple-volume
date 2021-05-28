#pragma once

#include <nanovdb/NanoVDB.h>

struct Scene {
  int frame_width;
  int frame_height;
  const nanovdb::FloatGrid *volume_grid;
  float light_dir[3];
  float light_half_angle;
  int spp;
  float max_value;
  struct PhaseFunction {
    float color[3];
  } phase_func;
};

void render(const Scene &scene, float *d_image);