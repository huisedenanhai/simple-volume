#pragma once

#include <nanovdb/NanoVDB.h>

struct Scene {
  int frame_width;
  int frame_height;
  const nanovdb::FloatGrid *volume_grid;
};

void render(const Scene &scene, float *d_image);