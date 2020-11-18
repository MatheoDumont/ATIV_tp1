/* hough_line.cpp
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 * Description :
 */
#include "hough_line.h"

HoughLine::HoughLine(cv::Mat im_threshold, int _n_theta, int _n_rho)
    : n_theta(_n_theta), n_rho(_n_rho)
{
    // theta \in [-pi/2,pi] (dtheta = 3pi/(2*N_theta))
    // rho   \in [0, sqrt(H*H+W*W)] (drho = sqrt(H*H+W*W)/N_rho)
    rows = im_threshold.rows;
    cols = im_threshold.cols;

    d_theta = (M_PI + M_PI_2) / float(n_theta);
    d_rho = sqrt(rows * rows + cols * cols) / float(n_rho);

    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            if (im_threshold.at<float>(row, col) > 0.5f)
                contours.push_back(Point(col, row, 0.0));

    //accumulator_vote_value = 1.0 / (contours.size());
    accumulator_vote_value = 2.0 / (contours.size() * (contours.size()-1));
    // is equal to 1 over the number of pairs of contours -> is robust against scaling
    accumulator = cv::Mat::zeros(n_theta, n_rho, CV_32F);
    // on prend des float directement, au moins on pourra mettre des demi-vote etc...
}

HoughLine::~HoughLine() {}
cv::Mat HoughLine::get_accumulator() { return accumulator; }

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

    accumulator.at<float>(idx_theta, idx_rho) += accumulator_vote_value;
}

Line_paremeters HoughLine::compute_line_parameters(Point i, Point j)
{
    float xi = i._x, yi = i._y;
    float xj = j._x, yj = j._y;

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
    // pour chaque couple de points du contour
    for (int i1 = 0; i1 < contours.size() - 1; i1++)
        for (int i2 = i1 + 1; i2 < contours.size(); i2++)
        {
            Line_paremeters line_param = compute_line_parameters(
                contours[i1], contours[i2]);
            update_accumulator(line_param);
        }
}

std::vector<Vote_paremeters> HoughLine::vote_threshold_local_maxima(float threshold, int radius)
{
    std::vector<Vote_paremeters> good_lines;

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
                    if (!to_keep)
                    {
                        break;
                    }
                }
                if (to_keep)
                {
                    good_lines.push_back(
                        Vote_paremeters({thet * d_theta - M_PI_2, rho * d_rho}));
                }
            }
        }
    }
    return good_lines;
}

/*----------------------display------------------------- */
bool critere(Point u, float epsilon, float tandir, float invtandir)
{
    return ((abs(u._y - u._x * tandir) < epsilon ||
    abs(u._x - u._y * invtandir) < epsilon));
}

cv::Mat HoughLine::line_display_image(std::vector<Vote_paremeters> lines)
{
    float epsilon_rad = d_theta * 0.5; // in radian
    float epsilon_pix = 0.7;            // in pixel
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_32F);
    for (int i = 0; i < contours.size(); i++)
    {
        img.at<float>(contours[i]._y, contours[i]._x) = 0.5;
    }

    for (int i = 0; i < lines.size(); i++)
    {
        float xh = lines[i].second * cos(lines[i].first); // xh  = rho * cos(theta)
        float yh = lines[i].second * sin(lines[i].first); // yh  = rho * sin(theta)
        float line_direction_0 = lines[i].first + M_PI_2;
        /*
        //critère 1
        if (line_direction_0 > M_PI)
            line_direction_0 -= 2*M_PI; // to have it in [-pi, pi] to compare to atan2
        float line_direction_1 = lines[i].first-M_PI_2;// should be in [-pi, pi] since theta in [-pi/2,pi]
        */

        //critère2
        float tandir = tan(lines[i].first + M_PI_2);
        float invtandir = 1. / tandir;

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                float y = row - yh;
                float x = col - xh;

                /*
                // critère 1
                if (abs(std::atan2(y,x) - line_direction_0) < epsilon_rad || abs(std::atan2(y,x) - line_direction_1) < epsilon_rad )
                    img.at<float>(row, col) += 0.3;
                */

                // critère 2
                if (abs(y - x * tandir) < epsilon_pix || abs(x - y * invtandir) < epsilon_pix)
                    img.at<float>(row, col) += 0.2;
            }
        }
    }
    return img;
}

cv::Mat HoughLine::segment_display_image(std::vector<Vote_paremeters> lines)
{// doesn't really work...
    float epsilon_rad = d_theta * 0.5; // in radian
    float epsilon_pix = 0.7;            // in pixel
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_32F);
    for (int i = 0; i < contours.size(); i++)
    {
        img.at<float>(contours[i]._y, contours[i]._x) = 0.0;
    }


    for (int i = 0; i < lines.size(); i++)
    {
        Point h = Point(lines[i].second * cos(lines[i].first),lines[i].second * sin(lines[i].first), 0.0);
        // xh  = rho * cos(theta)
        // yh  = rho * sin(theta)
        float line_direction_0 = lines[i].first + M_PI_2;

        //use critere
        float tandir = tan(lines[i].first + M_PI_2);
        float invtandir = 1. / tandir;


        float row_beg = 0., col_beg = 0.;
        float row_end = rows-1, col_end = cols-1;
        int j = 0;
        while (j < contours.size() &&
               !critere(contours[j] - h, epsilon_pix, tandir, invtandir))
        {j++;} // find first point of the line in the contour
        if (j < contours.size())
        { row_beg = contours[j]._y; col_beg = contours[j]._x; }

        j = contours.size()-1;
        while (j >= 0 &&
               !critere(contours[j] - h, epsilon_pix, tandir, invtandir))
        {j--;} // find last point of the line in the contour
        if (j >= 0)
        { row_end = contours[j]._y; col_end = contours[j]._x;}

        //std::cout << i <<" : DEBEND : " << row_beg << " " << row_end << " " << col_beg << " " << col_end << "\n";

        int row_init = std::max((int)std::min(row_beg, row_end)-1,0);
        int col_init = std::max((int)std::min(col_beg, col_end)-1,0);
        int row_fina = std::min((int)std::max(row_beg, row_end)+1,rows-1);
        int col_fina = std::min((int)std::max(col_beg, col_end)+1,cols-1);
        //std::cout << "  : INIFIN : " << row_init << " " << row_fina << " " << col_init << " " << col_fina << "\n";

        // begin images loop in rectangle {row_beg,row_end,col_beg,col_end}
        for (int row = row_init; row < row_fina+1; row++)
        {
            for (int col = col_init; col < col_fina+1; col++)
            {
                Point u = Point(1.0*col, 1.0*row, 0.0) - h;
                // critère
                if (critere(u, epsilon_pix, tandir, invtandir))
                    img.at<float>(row, col) += 0.5;
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
