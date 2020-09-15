#include "kernel.h"

#include <opencv4/opencv2/core.hpp>
// #include <opencv4/opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>
#include <vector>
#include <assert.h> /* assert */
#include <cmath>    /* sqrt atan2 */

void Kernel::conv_pixel(cv::Mat &in, cv::Mat &out, int row, int col, cv::Mat filtre)
{
  // cv::cvtColor(image, cv::COLOR_BGR2GRAY);
  // on ne conv pas sur les bordures de l'image im
  assert(row > 0 && row < in.rows - 1 && col > 0 && col < in.cols - 1);

  // filtre carre
  int step = floor(filtre.rows / 2);

  for (int i = -step; i < step + 1; i++)
  {
    for (int j = -step; j < step + 1; j++)
    {
      // float coef_in = (float)(in.at<uchar>(cv::Point(row + i, col + j)));
      // float coef_filtre = (filtre.at<float>(cv::Point(i + 1, j + 1)));
      out.at<float>(row, col) += (float)(in.at<uchar>(row + i, col + j)) * filtre.at<float>(i + 1, j + 1);
    }
  }
}

std::vector<cv::Mat> Kernel::conv2(cv::Mat &image, std::vector<cv::Mat> filtres)
{

  //  outputFrame.at<cv::Vec3b>(i, j) = final_region[regioned[i][j]];
  std::vector<cv::Mat> output;

  for (int i = 0; i < filtres.size(); i++)
  {
    output.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32F));
  }

  for (int row = 1; row < image.rows - 1; row++)
  {
    for (int col = 1; col < image.cols - 1; col++)
    {
      for (int f = 0; f < filtres.size(); f++)
      {
        conv_pixel(image, output[f], row, col, filtres[f]);
      }
    }
  }
  return output;
}

cv::Mat Kernel::amplitude_x(std::vector<cv::Mat> mi, float x)
{
  assert(mi.size() > 0);
  float temporary = 0.0;
  cv::Mat res = cv::Mat::zeros(mi[0].rows, mi[0].cols, CV_32F);
  for (size_t row = 0; row < mi[0].rows; row++)
  {
    for (size_t col = 0; col < mi[0].cols; col++)
    {
      temporary = 0.0;
      for (size_t i = 0; i < mi.size(); i++)
      {
        temporary += pow(abs(mi[i].at<float>(row, col)), x);
      }
      res.at<float>(row, col) = pow(1.0 / mi.size() * temporary, 1 / x);
    }
  }
  return res;
}

cv::Mat Kernel::amplitude_0(std::vector<cv::Mat> mi)
{
  assert(mi.size() > 0);
  float temporary = 0.0;
  cv::Mat res = cv::Mat::zeros(mi[0].rows, mi[0].cols, CV_32F);
  for (size_t row = 0; row < mi[0].rows; row++)
  {
    for (size_t col = 0; col < mi[0].cols; col++)
    {
      for (size_t i = 0; i < mi.size(); i++)
      { // loop in order to get the max
        temporary = abs(mi[i].at<float>(row, col));
        if (res.at<float>(row, col) < temporary)
        {
          res.at<float>(row, col) = temporary;
        }
      }
    }
  }
  return res;
}

cv::Mat Kernel::amplitude_1(std::vector<cv::Mat> mi)
{
  assert(mi.size() > 0);
  // We could just call amplitude_x(mi, 1.0)
  cv::Mat res = cv::Mat::zeros(mi[0].rows, mi[0].cols, CV_32F);
  for (size_t row = 0; row < mi[0].rows; row++)
  {
    for (size_t col = 0; col < mi[0].cols; col++)
    {
      for (size_t i = 0; i < mi.size(); i++)
      { // loop for adding the absolute values
        res.at<float>(row, col) += abs(mi[i].at<float>(row, col));
      }
      res.at<float>(row, col) = 1.0 / (mi.size()) * res.at<float>(row, col);
    }
  }
  return res;
}

cv::Mat Kernel::amplitude_2(std::vector<cv::Mat> mi)
{
  assert(mi.size() > 0);
  // We could just call amplitude_x(mi, 2.0)
  float temporary = 0.0;
  cv::Mat res = cv::Mat::zeros(mi[0].rows, mi[0].cols, CV_32F);
  for (size_t row = 0; row < mi[0].rows; row++)
  {
    for (size_t col = 0; col < mi[0].cols; col++)
    {
      temporary = 0.0;
      for (size_t i = 0; i < mi.size(); i++)
      { // loop in order to add the squares
        temporary += mi[i].at<float>(row, col) * mi[i].at<float>(row, col);
      }
      res.at<float>(row, col) = sqrt(1.0 / mi.size() * temporary);
    }
  }
  return res;
}

cv::Mat Kernel::angle(cv::Mat mx, cv::Mat my)
{
  assert(mx.rows == my.rows && my.cols == my.cols);
  cv::Mat res = cv::Mat::zeros(mx.rows, mx.cols, CV_32F);
  for (size_t row = 0; row < mx.rows; row++)
  {
    for (size_t col = 0; col < mx.cols; col++)
    {
      //   res.at<float>(row, col) = atan2( my.at<float>(row, col) / mx.at<float>(row, col) );
      res.at<float>(row, col) = atan2(my.at<float>(row, col), mx.at<float>(row, col));
    }
  }
  return res;
}
