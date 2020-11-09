#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <math.h>
#include "kernel.h"
#include "seuil.h"
#include "path_contour.h"
#include "contour.h"

#define IMAGE_NAME0 "datas/square_sample.png"
#include "hough_line.h"

int main(int argc, char *argv[])
{
  cv::Mat horizontal = (cv::Mat_<float>(3, 3) << -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f);
  cv::Mat vertical = (cv::Mat_<float>(3, 3) << 1 / 3.f, 1 / 3.f, 1 / 3.f, 0.f, 0.f, 0.f, -1 / 3.f, -1 / 3.f, -1 / 3.f);
  cv::Mat quart_plus = (cv::Mat_<float>(3, 3) << 1.f / 3.f, 1.f / 3.f, 0.f, 1.f / 3.f, 0.f, -1.f / 3.f, 0.f, -1.f / 3.f, -1.f / 3.f);
  cv::Mat quart_moins = (cv::Mat_<float>(3, 3) << 0.f, 1.f / 3.f, 1.f / 3.f, -1.f / 3.f, 0.f, 1.f / 3.f, -1.f / 3.f, -1.f / 3.f, 0.f);

  std::vector<cv::Mat> gradient_filters;
  gradient_filters.push_back(horizontal);
  gradient_filters.push_back(quart_moins);
  gradient_filters.push_back(vertical);
  gradient_filters.push_back(quart_plus);

  cv::Mat greyscale_image;
  cv::Mat im;
  if (argc < 2)
      im = cv::imread(IMAGE_NAME0);
  else
      im = cv::imread(argv[1]);

  cv::cvtColor(im, greyscale_image, cv::COLOR_BGR2GRAY);
  greyscale_image.convertTo(greyscale_image, CV_32F);

  cv::Mat amp0;
  cv::Mat dir;

  std::vector<cv::Mat> gradient_convol = Kernel::conv2(greyscale_image, gradient_filters);

  amp0 = Kernel::amplitude_0(gradient_convol);
  dir = Kernel::angle(gradient_convol);

  cv::Mat im_threshold = Seuil::seuil_global(amp0 * (1 / 255.0), 0.1);

  HoughLine houghline(im_threshold, 7, 4);
  houghline.compute_accumulator();
  std::vector<Line_paremeters> lines = houghline.vote_threshold_local_maxima(100);
  cv::Mat display = houghline.line_display_image(lines);

  cv::imshow("Lines in im", display);
  cv::waitKey(0);

  return 0;
}
