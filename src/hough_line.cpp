/* hough.cpp
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 * Description :
 */
#include "hough_line.h"

HoughLine::HoughLine(cv::Mat image, int _n_theta, int _n_rho) : image(image), n_theta(_n_theta), n_rho(_n_rho)
{
    // theta \in [-pi/2,pi] (dtheta = 3pi/(2*N_theta))
    // rho   \in [0, sqrt(H*H+W*W)] (drho = sqrt(H*H+W*W)/N_rho)

    d_theta = (M_PI + M_PI_2) / float(n_theta);
    d_rho = sqrt(image.rows * image.rows + image.cols * image.cols) / float(n_rho);
    accumulator = cv::Mat::zeros(n_theta, n_rho, CV_32F);
    // on prend des float directement, au moins on pourra mettre des demi-vote etc...
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

    accumulator.at<float>(idx_theta, idx_rho) += 1.;
}

Line_paremeters HoughLine::compute_line_parameters(Point i, Point j)
{
    float xi = i.first, yi = i.second;
    float xj = j.first, yj = j.second;

    float rho = abs(xi * yj - xj * yi) / sqrt(pow(xj - xi, 2) + pow(yj - yi, 2));

    float theta = std::atan((xi - xj) / (yj - yi));
    // pour garder theta dans l'intervalle [3pi/2, pi]
    if (theta > M_PI && theta < (M_PI + M_PI_2))
        theta -= M_PI;

    if (theta < 0)
        theta += 2 * M_PI;
    /* j'ai un doute pour ton theta
    * Par exemple si on prend i = (0,1) j = (1,2) ça donne le même theta
    * que i = (0,-1) j = (0,1), il faut faire gaffe aux signes
    */
    return Line_paremeters({theta, rho});

    // ma version
    float line_direction_norm = sqrt(pow(xj - xi, 2) + pow(yj - yi, 2))
    float signedrho = (xi * yj - xj * yi) / line_direction_norm;
    float xh = signedrho * (yj-yi)/line_direction_norm;
    float yh = signedrho * (xj-xi)/line_direction_norm;
    float theta = std::atan2(yh,xh);
    //https://en.cppreference.com/w/cpp/numeric/math/atan2
    // atan2(y,x) -> angle of (x,y) in [-pi,pi]
    return Line_paremeters({theta, abs(signedrho)});
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
