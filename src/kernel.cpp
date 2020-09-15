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
    // const uchar *irow_in = in.ptr(row + i);
    // const uchar *irow_filtre = filtre.ptr(i + 1);
    // const uchar *irow_out = out.ptr(row);

    for (int j = -step; j < step + 1; j++)
    {
      // float coef_in = ((const float *)irow_in)[col + j];
      // float coef_in = (float *)in.ptr(row + i, col + j);

      float coef_in = (float)(in.at<uchar>(cv::Point(row + i, col + j)));
      float coef_filtre = (filtre.at<float>(cv::Point(i + 1, j + 1)));
      // float coef_filtre = ((const float *)irow_filtre)[j + 1];
      float coef_res = coef_in * coef_filtre;

      // out.at<float>(row, col) += in.at<float>(row + i, col + j) * filtre.at<float>(i + 1, j + 1);
      // ((const float *)irow_out)[col] = coef_res;
      out.at<float>(row, col) += coef_res;
    }
  }
}

std::vector<cv::Mat> Kernel::conv2(cv::Mat &image, std::vector<cv::Mat> filtres)
{

  //  outputFrame.at<cv::Vec3b>(i, j) = final_region[regioned[i][j]];
  std::vector<cv::Mat> output;

  for (size_t i = 0; i < filtres.size(); i++)
  {
    output.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32F));
  }

  for (size_t row = 1; row < image.rows - 1; row++)
  {
    for (size_t col = 1; col < image.cols - 1; col++)
    {
      for (size_t f = 0; f < filtres.size(); f++)
      {
        conv_pixel(image, output[f], row, col, filtres[f]);
      }
    }
  }
  return output;
}

// cv::saturate_cast<uchar> (double)

cv::Mat amplitude_0(cv::Mat mx, cv::Mat my)
{
  assert(mx.rows == my.rows && my.cols == my.cols);
  float sumx = 0.0;
  float sumy = 0.0;
  cv::Mat res = cv::Mat::zeros(mx.rows, mx.cols, CV_32F);
  for (size_t row = 0; row < mx.rows; row++)
  {
    for (size_t col = 0; col < mx.cols; col++)
    {
      sumx = mx.at<float>(row, col);
      sumy = my.at<float>(row, col);
      res.at<float>(row, col) = 1 / (sqrt(2)) * sqrt(sumx * sumx + sumy * sumy);
    }
  }
  return res;
}

// cv::Mat amplitude_1(std::vector<cv::Mat> mi) {
//   assert(mx.rows == my.rows && my.cols == my.cols);
//   float temporary = 0.0;
//   cv::Mat res = cv::Mat::zeros(mx.rows, mx.cols, CV_32F);
//   for (size_t row = 0 ; row < mx.rows; row++) {
//     for (size_t col = 0; col < mx.cols; col++) {
//       for (size_t i ; i<mi.size() ; i++) { // loop in order to get the max
//         temporary =  abs(mi[i].at<float>(row, col));
//         if (res.at<float>(row, col) < temporary)
//           res.at<float>(row, col) = temporary;
//       }
//     }
//   }
//   return res;
// }

cv::Mat angle_0(cv::Mat mx, cv::Mat my)
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
