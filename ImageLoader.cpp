#include "ImageLoader.h"

#include <iostream>
#include <vector>

#include "ThirdParty/stb_image/stb_image.h"

std::vector<double> loadImage(const std::string& imagePath)
{
  int width, height, colorChannels;

  unsigned char* data = stbi_load(imagePath.c_str(), &width, &height, &colorChannels, 1);

  if (!data)
  {
    std::cerr << "Failed to load image\n";
    return {};
  }

  auto imageSize = width * height;
  std::vector<double> pixels(imageSize);
  for (int pixelIdx = 0; pixelIdx < imageSize; pixelIdx++)
  {
    pixels[pixelIdx] = data[pixelIdx] / 255.0;
  }

  stbi_image_free(data);

  return pixels;
}
