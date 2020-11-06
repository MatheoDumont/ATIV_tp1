/* hough.cpp
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 * Description :
 */
#include "hough_line.h"

HoughLine::HoughLine(cv::Mat image, int _n_theta, int _n_rho) : image(image), n_theta(_n_theta), n_rho(_n_rho)
{
    d_theta = (M_PI + M_PI_2) / float(n_theta);
    d_rho = sqrt(image.rows * image.rows + image.cols * image.cols) / float(n_rho);
    accumulator = cv::Mat::zeros(n_theta, n_rho, CV_32S);
}

HoughLine::~HoughLine() {}

void HoughLine::vote_accumulator(float rho, float theta)
{

    // https://en.cppreference.com/w/cpp/numeric/math/round
    // round a l'air de faire le job, mais peut surement causer des problemes
    // en terme de distribution (2.5 => 3)

    int tmp_theta = std::round((theta + M_PI_2) / d_theta);
    int tmp_rho = std::round((rho / d_rho));

    // since C++17
    // int idx_theta = std::clamp(tmp_theta, 0, n_theta);
    // int idx_rho = std::clamp(tmp_rho, 0, n_rho);
    int idx_theta, idx_rho;

    if (tmp_theta < 0)
        idx_theta = 0;
    else if (tmp_theta > n_theta - 1)
        idx_theta = n_theta - 1;
    else
        idx_theta = tmp_theta;

    if (tmp_rho < 0)
        idx_rho = 0;
    else if (tmp_rho > n_rho - 1)
        idx_rho = n_rho - 1;
    else
        idx_rho = tmp_rho;

    accumulator.at<int>(idx_theta, idx_rho) += 1;
}

// cv::Mat Hough::accumulator_line(cv::Mat image, int N_rho, int N_theta)
// {
//     // int rows = ceil((M_PI + M_PI_2) / float(N_theta));
//     // int cols = ceil(sqrt(image.rows * image.rows + image.cols * image.cols));
//     return cv::Mat::zeros(N_theta, N_rho, CV_32S);
// }

// void Hough::vote_accumulator_lines(cv::Mat &accumulator, float rho, float theta)
// {

// }