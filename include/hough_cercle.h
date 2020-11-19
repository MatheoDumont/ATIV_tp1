#ifndef HOUGH_CERCLE_H
#define HOUGH_CERCLE_H

#include <opencv4/opencv2/core.hpp>
#include "math.h"
#include "point.h"
#include <vector>

// std::tuple<x, y, radius>, parameters of the cercle, (x, y) is the center.
typedef std::tuple<float, float, float> Cercle_parameters;

class HoughCercle
{
private:
    std::vector<Point> contours;

    int rows, cols;
    int n_x, n_y, n_r;
    float rad_min, rad_max;
    float d_x, d_y, d_r;
    float accumulator_vote_value;

public:
    cv::Mat accumulator;

    HoughCercle(
        cv::Mat im_threshold,
        float _rad_min, float _rad_max,
        int _n_x, int _n_y, int _n_r);
    // ~HoughCercle();

    Point circumscribed_triangle_circle(Point x, Point y, Point z);
    Cercle_parameters compute_cercle_parameters(Point x, Point y, Point z);
    void update_accumulator(Cercle_parameters cercle_param);
    void compute_accumulator();
    std::vector<Cercle_parameters> vote_threshold_local_maxima(float threshold, int radius);
    cv::Mat cercle_display_image(std::vector<Cercle_parameters> cercles);
};

#endif
