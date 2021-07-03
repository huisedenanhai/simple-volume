#include <iostream>
#include <stb_image.h>
#include <string>

struct Image {
  float *data;
  int w, h, c;

  Image(const std::string &name) {
    data = stbi_loadf(name.c_str(), &w, &h, &c, 0);
    if (data == nullptr) {
      throw std::runtime_error("failed to open image " + name);
    }
  }

  ~Image() {
    stbi_image_free(data);
  }
};

float calc_mse(const Image &img1, const Image &img2) {
  if (img1.w != img2.w || img1.h != img2.h || img1.c != img2.c) {
    throw std::runtime_error("image size mismatch");
  }
  float acc = 0.0f;
  int count = img1.w * img1.h * img1.c;
  for (int i = 0; i < count; i++) {
    float delta = img1.data[i] - img2.data[i];
    acc += delta * delta;
  }
  return acc / float(count);
}

int main() {
  std::string scenes[] = {
      "bunny_cloud",
      "smoke2",
  };
  std::string methods[] = {
      "dt",
      "rt",
      "mis",
  };
  std::cout << "| |";
  for (int i = 0; i < std::size(methods); i++) {
    std::cout << methods[i] << " |";
  }
  std::cout << std::endl;

  std::cout << "|";
  for (int i = 0; i <= std::size(methods); i++) {
    std::cout << "---|";
  }
  std::cout << std::endl;

  for (const auto &scene : scenes) {
    auto get_name = [&](const std::string &m) {
      return "result/" + scene + "_" + m + "/output.hdr";
    };
    std::cout << "|" << scene << "|";
    Image ref(get_name("ref"));
    for (const auto &m : methods) {
      Image img(get_name(m));
      auto mse = calc_mse(ref, img);
      std::cout << std::scientific << mse << "|";
    }
    std::cout << std::endl;
  }
}