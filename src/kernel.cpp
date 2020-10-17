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


int Kernel::direction_from_vec(int row, int col)
{
		// be careful : we use visual angles :
		/*  3	2	1
				4	x	0
				5	6	7 */
	if (row == 0 && col >  0)
		return 0;
	if (row <  0 && col >  0)
		return 1;
	if (row <  0 && col == 0)
		return 2;
	if (row <  0 && col <  0)
		return 3;
	if (row == 0 && col <  0)
		return 4;
	if (row >  0 && col <  0)
		return 5;
	if (row >  0 && col == 0)
		return 6;
	if (row >  0 && col >  0)
		return 7;
	return -1; // case (0,0)
}

std::pair<int,int> Kernel::vec_from_direction(int dir)
{
	switch (dir) {
		case 0:
			return {0,1};
		break;
		case 1:
			return {-1,1};
		break;
		case 2:
			return {-1,0};
		break;
		case 3:
			return {-1,-1};
		break;
		case 4:
			return {0,-1};
		break;
		case 5:
			return {1,-1};
		break;
		case 6:
			return {1,0};
		break;
		case 7:
			return {1,1};
		break;
		default:
			return {0,0};
		break;
	}
}

float Kernel::conv_pixel(cv::Mat &in, int row, int col, cv::Mat filtre)
{
  // cv::cvtColor(image, cv::COLOR_BGR2GRAY);
  // on ne conv pas sur les bordures de l'image im
  assert(row > 0 && row < in.rows - 1 && col > 0 && col < in.cols - 1);

  // filtre carre
  int step = floor(filtre.rows / 2);
  float output = 0.f;
  for (int i = -step; i < step + 1; i++)
  {
    for (int j = -step; j < step + 1; j++)
    {
      // float coef_in = (float)(in.at<uchar>(cv::Point(row + i, col + j)));
      // float coef_filtre = (filtre.at<float>(cv::Point(i + 1, j + 1)));
      // out.at<float>(row, col) += (float)(in.at<uchar>(row + i, col + j)) * filtre.at<float>(i + 1, j + 1);
      output += in.at<float>(row + i, col + j) * filtre.at<float>(i + 1, j + 1);
    }
  }
  return output;
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
        output[f].at<float>(row, col) = conv_pixel(image, row, col, filtres[f]);
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


cv::Mat Kernel::angle(std::vector<cv::Mat> mi)
{
	assert(mi.size() > 0);
	size_t i, imax;
	size_t n = mi.size();
	float maxi = 0.0;
	float temp = 0.0;
	cv::Mat res = cv::Mat::zeros(mi[0].rows, mi[0].cols, CV_32F);
  for (size_t row = 0; row < mi[0].rows; row++)
  {
    for (size_t col = 0; col < mi[0].cols; col++)
    {
			maxi = 0.0;
			imax = 0;
      for (i = 0; i < n; i++)
      { // to find the gradient that has the maximum absolute value
				temp = abs(mi[i].at<float>(row, col));
        if (temp > maxi)
				{
					imax = i;
					maxi = temp;
				}
      }
			if (mi[imax].at<float>(row, col) < 0.0) // gradient in opposite direction
      	res.at<float>(row, col) = 1.0 * (imax + n);
			else // classical gradient directions
      	res.at<float>(row, col) = 1.0 * imax;
    }
  }
  return res;
}

cv::Mat Kernel::angle_arctan(cv::Mat mx, cv::Mat my)
{// angle in [0, 2*M_PI]
  assert(mx.rows == my.rows && my.cols == my.cols);
  cv::Mat res = cv::Mat::zeros(mx.rows, mx.cols, CV_32F);
  float temporary;
  for (size_t row = 0; row < mx.rows; row++)
  {
    for (size_t col = 0; col < mx.cols; col++)
    {
      temporary = atan2(my.at<float>(row, col), mx.at<float>(row, col));
      if (temporary < 0) // in order to have the same angle but in [0,2pi]
				temporary += 2*M_PI;
      res.at<float>(row, col) = temporary;
    }
  }
  return res;
}

void HSVtoRGB(float H, float S,float V, cv::Vec3f& pixel){
    if(H>360 || H<0 || S>255 || S<0 || V>255 || V<0){
      std::cout<<"The given HSV values are not in valid range"<<std::endl;
      return;
    }
    float s = S/255;
    float v = V/255;
    float C = s*v;
    float X = C*(1-abs(fmod(H/60.0, 2)-1));
    float m = v-C;
    float r,g,b;
    if(H >= 0 && H < 60){
        r = C,g = X,b = 0;
    }
    else if(H >= 60 && H < 120){
        r = X,g = C,b = 0;
    }
    else if(H >= 120 && H < 180){
        r = 0,g = C,b = X;
    }
    else if(H >= 180 && H < 240){
        r = 0,g = X,b = C;
    }
    else if(H >= 240 && H < 300){
        r = X,g = 0,b = C;
    }
    else{
        r = C,g = 0,b = X;
    }
    pixel[0] = (r+m)*255; // red
    pixel[1] = (g+m)*255; // green
    pixel[2] = (b+m)*255; // blue
}

cv::Mat Kernel::color_gradient_im(cv::Mat amp, cv::Mat ang)
{
  assert(amp.rows == ang.rows && amp.cols == ang.cols);
  cv::Mat img=cv::Mat::zeros(amp.rows,amp.cols,CV_32FC3);
  for (int row = 0; row < img.rows; row++)
    {
    for (int col = 0; col < img.cols; col++)
      {
	HSVtoRGB(ang.at<float>(row, col)*180.0/M_PI,
		 255.,
		 amp.at<float>(row, col),
		 img.at<cv::Vec3f>(row,col));
      }
    }
  return img;
}
