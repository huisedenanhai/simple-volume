#include "render.h"
#include <chrono>
#include <config.h>
#include <cpptoml.h>
#include <filesystem>
#include <iostream>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/io/Stream.h>
#include <openvdb/openvdb.h>
#include <stb_image_write.h>
#include <stdexcept>

static void save_result(const std::filesystem::path &dir,
                        float *d_image,
                        int width,
                        int height) {
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
  std::filesystem::create_directories(dir);
  stbi_flip_vertically_on_write(1);
  auto hdr_path = dir / "output.hdr";
  stbi_write_hdr(hdr_path.string().c_str(), width, height, 3, result.data());
  auto png_path = dir / "output.png";
  stbi_write_png(
      png_path.string().c_str(), width, height, 3, resultAsBytes.data(), 0);
  std::cout << "result written to " << dir << std::endl;
}

int main(int argc, char **argv) {
  std::cout << DATA_DIR << std::endl;
  openvdb::initialize();
  if (argc < 2) {
    throw std::runtime_error("please specify input");
  }
  std::string config_file = argv[1];
  std::cout << "load config " << config_file << std::endl;
  auto config = cpptoml::parse_file(config_file);

  std::ifstream is(*config->get_as<std::string>("file"), std::ios_base::binary);
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

  int frame_width = config->get_as<int>("frame_width").value_or(800);
  int frame_height = config->get_as<int>("frame_height").value_or(600);
  size_t image_buffer_size = frame_width * frame_height * 3 * sizeof(float);
  float *d_image;
  cudaMalloc((void **)(&d_image), image_buffer_size);
  cudaMemset(d_image, 0, image_buffer_size);

  auto get_config_float3 =
      [&](float *res, const std::string &key, float v0, float v1, float v2) {
        auto arr = config->get_qualified_array_of<double>(key);
        if (!arr || arr->size() < 3) {
          res[0] = v0;
          res[1] = v1;
          res[2] = v2;
        } else {
          for (int i = 0; i < 3; i++) {
            res[i] = arr->at(i);
          }
        }
      };

  Scene scene{};
  scene.frame_width = frame_width;
  scene.frame_height = frame_height;
  scene.volume_grid = d_grid;
  get_config_float3(scene.camera_pos, "camera_pos", 0.0f, 20.0f, 100.0f);
  get_config_float3(scene.light_dir, "light_dir", 0.0f, 1.0f, 0.0f);
  get_config_float3(scene.light_color, "light_color", 1.0f, 1.0f, 1.0f);
  scene.light_cos_angle = static_cast<float>(
      config->get_as<double>("light_cos_angle").value_or(-1.0));
  scene.spp = config->get_as<int>("spp").value_or(10);

  float min_value, max_value;
  grid->evalMinMax(min_value, max_value);
  printf("min: %f, max: %f\n", min_value, max_value);

  scene.max_value = max_value;

  get_config_float3(scene.phase_func.color, "color", 0.8f, 0.8f, 0.8f);
  scene.phase_func.cos_angle = static_cast<float>(
      config->get_as<double>("phase_cos_angle").value_or(-1.0));
  scene.phase_func.diffuse =
      static_cast<float>(config->get_as<double>("phase_diffuse").value_or(1.0));

  scene.mode =
      render_mode_from_str(config->get_as<std::string>("mode").value_or(""));

  auto start_time = std::chrono::system_clock::now();
  {
    auto t = std::chrono::system_clock::to_time_t(start_time);
    std::cout << "render start at " << std::ctime(&t) << std::flush;
  }

  render(scene, d_image);

  auto end_time = std::chrono::system_clock::now();
  {
    using namespace std::chrono;
    auto t = system_clock::to_time_t(end_time);
    // TODO more pretty output
    auto s = duration<float>(end_time - start_time).count();
    auto m = floor(s / 60.0);
    s -= m * 60.0f;
    auto h = floor(m / 60.0);
    m -= h * 60.0f;
    std::cout << "render finished at " << std::ctime(&t) << std::flush;
    std::cout << "duration " << h << ":" << m << ":" << s << std::endl;
  }

  save_result(config->get_as<std::string>("out").value_or("result"),
              d_image,
              frame_width,
              frame_height);
  cudaFree(d_image);
}