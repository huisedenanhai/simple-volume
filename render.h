#pragma once

#include <nanovdb/NanoVDB.h>

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
};

void render(const Scene &scene, float *d_image);