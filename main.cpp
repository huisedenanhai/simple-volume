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
  stbi_write_png("result/output.png", width, height, 3, resultAsBytes.data(), 0);
  printf("result written to result/output.png\n");
}

int main() {
  std::cout << DATA_DIR << std::endl;
  openvdb::initialize();
  try {
    std::ifstream is(DATA_DIR "/../.vscode/bunny_cloud.vdb",
                     std::ios_base::binary);
    auto grids = openvdb::io::Stream(is).getGrids();
    auto handle =
        nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(grids->at(0));
    const nanovdb::GridMetaData *metadata = handle.gridMetaData();
    if (metadata->gridType() != nanovdb::GridType::Float) {
      throw std::runtime_error("only support float grid");
    }
    handle.deviceUpload();
    const nanovdb::FloatGrid *grid = handle.deviceGrid<float>();

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
    scene.volume_grid = grid;

    render(scene, d_image);

    save_result(d_image, frame_width, frame_height);
    cudaFree(d_image);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}