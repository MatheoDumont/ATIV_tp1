/* hough.cpp
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 * Description :
 */
#include "hough.h"

cv::Mat Hough::accumulator_line(cv::Mat image, int N_rho, int N_theta)
{
    int rows = ceil((M_PI + M_PI_2) / float(N_theta));
    int cols = ceil(sqrt(image.rows * image.rows + image.cols * image.cols));
    return cv::Mat::zeros(rows, cols, CV_32S);
}

void Hough::vote_accumulator_lines(cv::Mat &accumulator, float rho, float theta)
{

}