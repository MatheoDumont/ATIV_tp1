/* hough.cpp
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 * Description :
 */
#include "hough_line.h"

HoughLine::HoughLine(cv::Mat _im_threshold, int _n_theta, int _n_rho) : im_threshold(_im_threshold), n_theta(_n_theta), n_rho(_n_rho)
{
    // theta \in [-pi/2,pi] (dtheta = 3pi/(2*N_theta))
    // rho   \in [0, sqrt(H*H+W*W)] (drho = sqrt(H*H+W*W)/N_rho)

    d_theta = (M_PI + M_PI_2) / float(n_theta);
    d_rho = sqrt(im_threshold.rows * im_threshold.rows + im_threshold.cols * im_threshold.cols) / float(n_rho);
    accumulator = cv::Mat::zeros(n_theta, n_rho, CV_32F);
    // on prend des float directement, au moins on pourra mettre des demi-vote etc...
}

HoughLine::~HoughLine() {}
cv::Mat HoughLine::get_accumulator() {return accumulator;}

//float rho, float theta
void HoughLine::update_accumulator(Line_paremeters line_param)
{

    // https://en.cppreference.com/w/cpp/numeric/math/round
    // round a l'air de faire le job, mais peut surement causer des problemes
    // en terme de distribution (2.5 => 3)

    int tmp_theta = std::round((line_param.first + M_PI_2) / d_theta);
    int tmp_rho = std::round((line_param.second / d_rho));

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

    // float rho = abs(xi * yj - xj * yi) / sqrt(pow(xj - xi, 2) + pow(yj - yi, 2));
    //
    // float theta = std::atan((xi - xj) / (yj - yi));
    // // pour garder theta dans l'intervalle [3pi/2, pi]
    // if (theta > M_PI && theta < (M_PI + M_PI_2))
    //     theta -= M_PI;
    //
    // if (theta < 0)
    //     theta += 2 * M_PI;
    // /* j'ai un doute pour ton theta
    // * Par exemple si on prend i = (0,1) j = (1,2) ça donne le même theta
    // * que i = (0,-1) j = (0,1), il faut faire gaffe aux signes
    // */
    // return Line_paremeters({theta, rho});

    float line_direction_norm = sqrt(pow(xj - xi, 2) + pow(yj - yi, 2));
    float signedrho = (xi * yj - xj * yi) / line_direction_norm;
    float xh = signedrho * (yj - yi) / line_direction_norm;
    float yh = signedrho * (xi - xj) / line_direction_norm;
    // h est le point de la droite le plus proche de l'origine (0,0)
    // ses coord polaires sont (rho, theta), ses coord cartésiennes sont (xh, yh)
    float theta = std::atan2(yh, xh);
    //https://en.cppreference.com/w/cpp/numeric/math/atan2
    // atan2(y,x) -> angle of (x,y) in [-pi,pi]
    return Line_paremeters({theta, abs(signedrho)});
}

void HoughLine::compute_accumulator()
{
    for (int i = 0; i < im_threshold.rows; i++)
    {
        for (int j = 0; j < im_threshold.cols; j++)
        {

            if (im_threshold.at<float>(i, j) > 0.5f)
            {

                for (int iprim = 0; iprim < im_threshold.rows; iprim++)
                {
                    for (int jprim = 0; jprim < im_threshold.cols; jprim++)
                    {

                        if ((iprim*im_threshold.cols+jprim > i*im_threshold.cols+j) && im_threshold.at<float>(iprim, jprim) > 0.5f)
                        {// la première condition est pour assurer que l'on passe une seule et une fois par paire
                            Line_paremeters line_param = compute_line_parameters({j, i}, {jprim, iprim});
                            update_accumulator(line_param);
                        }
                    }
                }
            }
        }
    }
}

std::vector<Line_paremeters> HoughLine::vote_threshold_local_maxima(float threshold, int radius)
{
    std::vector<Line_paremeters> good_lines;

    for (int thet = radius; thet < accumulator.rows - radius; thet++)
    {
        for (int rho = radius; rho < accumulator.cols - radius; rho++)
        {
            float vote_value = accumulator.at<float>(thet, rho);
            bool to_keep = true;
            if (vote_value > threshold)
            { // maybe a good line : verify if local maximum
                for (int i = -radius; i < radius + 1; i++)
                {
                    for (int j = -radius; j < radius + 1; j++)
                    {
                        if (accumulator.at<float>(thet + i, rho + j) > vote_value)
                        {
                            to_keep = false;
                            break;
                        }
                    }
                }
                if (to_keep)
                {
                    good_lines.push_back(
                        Line_paremeters({thet * d_theta - M_PI_2, rho * d_rho}));
                }
            }
        }
    }
    return good_lines;
}

cv::Mat HoughLine::line_display_image(std::vector<Line_paremeters> lines)
{
    float epsilon_rad = d_theta*0.5; // in radian
    float epsilon_pix = 1.; // in pixel
    cv::Mat img = cv::Mat::zeros(im_threshold.rows, im_threshold.cols, CV_32F);
    for (int row = 0; row < im_threshold.rows; row++)
    {
        for (int col = 0; col < im_threshold.cols; col++)
        {
            img.at<float>(row, col) = im_threshold.at<float>(row, col) * 0.0;
        }
    }

    for (int i = 0; i < lines.size(); i++)
    {
        float xh = lines[i].second * cos(lines[i].first); // xh  = rho * cos(theta)
        float yh = lines[i].second * sin(lines[i].first); // yh  = rho * sin(theta)
        float line_direction_0 = lines[i].first+M_PI_2;
        /*
        //critère 1
        if (line_direction_0 > M_PI)
            line_direction_0 -= 2*M_PI; // to have it in [-pi, pi] to compare to atan2
        float line_direction_1 = lines[i].first-M_PI_2;// should be in [-pi, pi] since theta in [-pi/2,pi]
        */

        //critère2
        float tandir = tan(lines[i].first+M_PI_2);
        float invtandir = 1./tandir;


        for (int row = 0; row < im_threshold.rows; row++)
        {
            for (int col = 0; col < im_threshold.cols; col++)
            {
                float y = row-yh;
                float x = col-xh;

                /*
                // critère 1
                if (abs(std::atan2(y,x) - line_direction_0) < epsilon_rad || abs(std::atan2(y,x) - line_direction_1) < epsilon_rad )
                    img.at<float>(row, col) += 0.3;
                */

                // critère 2
                if ( abs(y - x*tandir) < epsilon_pix || abs(x - y*invtandir) < epsilon_pix )
                    img.at<float>(row, col) += 0.3;

            }
        }
    }
    return img;
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
