#include "render.h"
#include <config.h>
#include <iostream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/io/Stream.h>
#include <openvdb/openvdb.h>
#include <stb_image_write.h>
#include <stdexcept>

static void save_result(float *d_image, int width, int height) {
  auto elemCnt = width * height * 3;
  std::vector<float> result(elemCnt);
  std::vector<unsigned char> resultAsBytes(elemCnt);
  cudaMemcpy((void *)result.data(),
             d_image,
             sizeof(float) * elemCnt,
             cudaMemcpyDeviceToHost);
  auto convertPixel = [](float v) {
    // convert to sRGB
    v = powf(v, 1.0f / 2.2f);
    v *= 255;
    if (v < 0) {
      v = 0;
    }
    if (v > 255) {
      v = 255;
    }
    return (unsigned char)(v);
  };
  auto toneMapping = [](float v) { return 1.0f - exp(-v); };
  for (int i = 0; i < elemCnt; i++) {
    resultAsBytes[i] = convertPixel(toneMapping(result[i]));
  }
  stbi_flip_vertically_on_write(1);
  stbi_write_hdr("result/output.hdr", width, height, 3, result.data());
  stbi_write_png(
      "result/output.png", width, height, 3, resultAsBytes.data(), 0);
  printf("result written to result/output.png\n");
}

int main() {
  std::cout << DATA_DIR << std::endl;
  openvdb::initialize();
  try {
    std::ifstream is(DATA_DIR
                     //  "/../.vscode/bunny_cloud.vdb"
                     //  "/../.vscode/smoke2.vdb"
                     "/smoke.vdb",
                     std::ios_base::binary);
    if (!is.good()) {
      throw std::runtime_error("failed to open file");
    }
    auto grids = openvdb::io::Stream(is).getGrids();
    auto grid = openvdb::GridBase::grid<openvdb::FloatGrid>(grids->at(0));

    auto handle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grid);
    const nanovdb::GridMetaData *metadata = handle.gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) {
      throw std::runtime_error("only support float grid");
    }
    handle.deviceUpload();
    const nanovdb::FloatGrid *d_grid = handle.deviceGrid<float>();

    auto bbox = handle.gridMetaData()->worldBBox();

    std::cout << "load success" << std::endl;

    int frame_width = 800;
    int frame_height = 600;
    size_t image_buffer_size = frame_width * frame_height * 3 * sizeof(float);
    float *d_image;
    cudaMalloc((void **)(&d_image), image_buffer_size);
    cudaMemset(d_image, 0, image_buffer_size);

    Scene scene{};
    scene.frame_width = frame_width;
    scene.frame_height = frame_height;
    scene.volume_grid = d_grid;
    scene.light_dir[0] = 0.0f;
    scene.light_dir[1] = 1.0f;
    scene.light_dir[2] = 0.0f;
    scene.spp = 100;

    float min_value, max_value;
    grid->evalMinMax(min_value, max_value);
    printf("min: %f, max: %f\n", min_value, max_value);

    scene.max_value = max_value;
    float color[] = {0.8, 0.8, 0.8};
    std::memcpy(scene.phase_func.color, color, sizeof(color));

    render(scene, d_image);

    save_result(d_image, frame_width, frame_height);
    cudaFree(d_image);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}