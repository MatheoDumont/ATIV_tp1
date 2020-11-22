#include "hough_cercle.h"

HoughCercle::HoughCercle(cv::Mat im_threshold,
                         float _rad_min, float _rad_max,
                         int _n_x, int _n_y, int _n_r)
    : rad_min(_rad_min), rad_max(_rad_max),
      n_x(_n_x), n_y(_n_y), n_r(_n_r)
{
    rows = im_threshold.rows;
    cols = im_threshold.cols; // saved for cercle display

    d_x = im_threshold.cols / (1.0 * n_x);
    d_y = im_threshold.rows / (1.0 * n_y);
    d_r = (rad_max - rad_min) / (1.0 * n_r);

    for (int row = 0; row < im_threshold.rows; row++)
        for (int col = 0; col < im_threshold.cols; col++)
            if (im_threshold.at<float>(col, row) > 0.5f)
                contours.push_back(Point(col, row, 0.0));

    accumulator_vote_value = 6.0 / (contours.size()*(contours.size()-1)*(contours.size()-2));
    int dims[] = {n_x, n_y, n_r};
    accumulator = cv::Mat(3, dims, CV_32F);
    //accumulator = cv::Mat(cv::Vec3i(n_x, n_y, n_r), CV_32F);
    std::cout << "accumulator size : "<< accumulator.size << "\n";
    std::cout << "contour size : " << contours.size() << "\n";
}

Point HoughCercle::circumscribed_triangle_circle(Point x, Point y, Point z)
{
    const Point &A = x;
    const Point &B = y;
    const Point &C = z;

    // angle A == CAB
    Point AB = B - A;
    Point AC = C - A;

    Point crossed = cross(AB, AC);
    float tan_A = norm(crossed) / dot(AB, AC);

    // angle B == ABC
    Point BA = A - B;
    Point BC = C - B;

    crossed = cross(BC, BA);
    float tan_B = norm(crossed) / dot(BC, BA);

    // angle C == BCA
    Point CB = B - C;
    Point CA = A - C;

    crossed = cross(CA, CB);
    float tan_C = norm(crossed) / dot(CA, CB);

    float tmp_alpha = tan_C + tan_B;
    float tmp_beta = tan_A + tan_C;
    float tmp_gamma = tan_B + tan_A;

    float alpha = tmp_alpha / (tmp_alpha + tmp_beta + tmp_gamma);
    float beta = tmp_beta / (tmp_alpha + tmp_beta + tmp_gamma);
    float gamma = tmp_gamma / (tmp_alpha + tmp_beta + tmp_gamma);

    Point centre;
    centre._x = A._x * alpha + B._x * beta + C._x * gamma;
    centre._y = A._y * alpha + B._y * beta + C._y * gamma;
    centre._z = A._z * alpha + B._z * beta + C._z * gamma;

    return centre;
}

Cercle_parameters HoughCercle::compute_cercle_parameters(Point x, Point y, Point z)
{
    Point centre = circumscribed_triangle_circle(x, y, z);
    float radius = norm(x - centre);

    return std::tuple<float, float, float>(centre._x, centre._y, radius);
}

int clamp(int min, int max, int val)
{
    if (val <= min)
        return min;
    else if (val >= max)
        return max;
    else
        return val;
}

void HoughCercle::update_accumulator(Cercle_parameters cercle_param)
{
    int idx_x = clamp(0, n_x-1, std::round(std::get<0>(cercle_param) / d_x));
    int idx_y = clamp(0, n_y-1, std::round(std::get<1>(cercle_param) / d_y));
    int idx_r = clamp(0, n_r-1, (std::round(std::get<2>(cercle_param)-rad_min) / d_r));

    accumulator.at<float>(idx_x, idx_y, idx_r) += accumulator_vote_value;
}

void HoughCercle::compute_accumulator()
{
    // pour chaque triplet de points du contour
    for (int i1 = 0; i1 < contours.size() - 2; i1++)
    {
        for (int i2 = i1 + 1; i2 < contours.size() - 1; i2++)
        {
            for (int i3 = i2 + 1; i3 < contours.size(); i3++)
            {
                Cercle_parameters cercle_param = compute_cercle_parameters(
                    contours[i1], contours[i2], contours[i3]);
                update_accumulator(cercle_param);
            }
        }
        std::cout << "compute : i1 = "<< i1 << "\n";
    }
    std::cout << "END compute Accumulator" << "\n";
}

std::vector<Cercle_parameters> HoughCercle::vote_threshold_local_maxima(float threshold, int radius)
{
    std::vector<Cercle_parameters> good_cercles;
    std::cout << "BEGIN Vote" << "\n";
    for (int ix = radius; ix < accumulator.size[0] - radius; ix++)
    {
        for (int iy = radius; iy < accumulator.size[1] - radius; iy++)
        {
            for (int ir = radius; ir < accumulator.size[2] - radius; ir++)
            {

                float vote_value = accumulator.at<float>(ix, iy, ir);
                bool to_keep = true;
                if (vote_value > threshold)
                { // maybe a good cercle : verify if local maximum
                    for (int ixp = -radius; ixp < radius + 1; ixp++)
                    {
                        for (int iyp = -radius; iyp < radius + 1; iyp++)
                        {
                            for (int irp = -radius; irp < radius + 1; irp++)
                            {
                                {
                                    if (accumulator.at<float>(ix + ixp, iy + iyp, ir + irp) > vote_value)
                                    {
                                        to_keep = false;
                                        break;
                                    }
                                }
                            }
                            if (!to_keep)
                            {
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
                        good_cercles.push_back(
                            Cercle_parameters({ix * d_x, iy * d_y, ir * d_r + rad_min}));
                    }
                }
            }

        }
        std::cout << "vote : ix = "<< ix << "\n";
    }
    std::cout << "END Vote" << "\n";
    return good_cercles;
}

cv::Mat HoughCercle::cercle_display_image(std::vector<Cercle_parameters> cercles)
{
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32F);

    for (int i = 0; i < cercles.size(); i++)
    {
        // parametres du cercles avec (x, y) le centre
        float x = std::get<0>(cercles[i]);
        float y = std::get<1>(cercles[i]);
        float radius = std::get<2>(cercles[i]);
        Point border(radius, 0., 0.);

        float step = 1.0/radius;

        for (float angle = 0; angle < 6.28; angle+=step)
        {
            int x_r = std::round(x + border._x);
            int y_r = std::round(y + border._y);
            if (x_r >= 0 && x_r < cols && y_r >= 0 && y_r < rows)
                 im.at<float>(x_r, y_r) += 0.4;
            border = rotation2D(border, step);
        }
    }

    return im;
}

cv::Mat HoughCercle::cercle_display_image_color(cv::Mat im_threshold, std::vector<Cercle_parameters> cercles)
{
    cv::Mat im = cv::Mat::zeros(rows, cols, CV_32FC3);
    for (int row = 0; row < im.rows; row++)
      {
      for (int col = 0; col < im.cols; col++)
        {
            im.at<cv::Vec3f>(row,col)[0]=0.4*im_threshold.at<float>(row,col);
        }
      }
    for (int i = 0; i < cercles.size(); i++)
    {
        // parametres du cercles avec (x, y) le centre
        float x = std::get<0>(cercles[i]);
        float y = std::get<1>(cercles[i]);
        float radius = std::get<2>(cercles[i]);
        Point border(radius, 0., 0.);

        float step = 1.0/radius;

        for (float angle = 0; angle < 6.28; angle+=step)
        {
            int x_r = std::round(x + border._x);
            int y_r = std::round(y + border._y);
            if (x_r >= 0 && x_r < cols && y_r >= 0 && y_r < rows)
            {
                im.at<cv::Vec3f>(x_r, y_r)[2] += 0.8;
                im.at<cv::Vec3f>(x_r, y_r)[1] += 0.8;
            }
            border = rotation2D(border, step);
        }
    }

    return im*255.;
}
