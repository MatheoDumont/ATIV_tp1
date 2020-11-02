/* hough.h
 * Authors : Yann-Situ GAZULL Matheo DUMONT
 */
#ifndef HOUGH_H
#define HOUGH_H

class Hough
{
private:
    Hough(/* args */);
    ~Hough();

public:
    //Plan :
    /* It would be interesting to create structure of Line (+ Circle and
    * Segment) pour éviter de se trimballer des paires de paires de floats ou
    * des vector de tailles 4 après le vote.
    *
    * On pourrait aussi y mettre en static les fonctions qui à deux points
    * associe la droite ou celle qui, en prenant en compte
    * (rho_max, N_rho, N_theta), met à jour l'accumulateur.
    *
    * Il faudrait faire plusieurs fichiers :
    * - accumulator.cpp/.h pour les accumulateurs.
    * - vote.cpp/.h pour les votes.
    * - et peut-être hough.cpp/.h pour Line, Circle, Segment et leurs
    *      accumulator_updates.
    */

    //=================
    /* Accumulators for lines.
    * For the moment, we can just add 1 to the closest discrete line.
    * A smarter approach is to add a convexe combination to the surrounding
    * discrete lines. -> todo after
    */

    /*
    * Compute accumulator of a line, with polar coordinates, from a
    * binary image.
    * Returns a matrice of size (N_rho,N_theta)
    * theta \in [-pi/2,pi] (dtheta = 3pi/(2*N_theta))
    * rho   \in [0, sqrt(H*H+W*W)] (drho = sqrt(H*H+W*W)/N_rho)
    */
    cv::Mat accumulator_line(cv::Mat image, int N_rho, int N_theta);

    /*
    * Compute accumulator of a line, with polar coordinates, from a list
    * of points.
    * Returns a matrice of size (N_rho,N_theta)
    * theta \in [-pi/2,pi] (dtheta = 3pi/(2*N_theta))
    * rho   \in [0, sqrt(H*H+W*W)] (drho = sqrt(H*H+W*W)/N_rho)
    */
    cv::Mat accumulator_line_list(std::vector<std::pair<int,int>> list_of_points,
        int N_rho, int N_theta,
        int H, int W);

    //=================
    /* Vote
    * For the moment we keep the lines that are local maximas above a threshold.
    */

    /*
    * Keep the lines that are local maximas above a threshold.
    */
    std::vector<std::pair<float,float>> vote_maxima_locaux(cv::Mat accumulator,
        float threshold);

    /*
    * Apply a mean filter on the accumulator and keep the lines that are local
    * maximas above a threshold.
    */
    std::vector<std::pair<float,float>> vote_maxima_locaux_mean(cv::Mat accumulator,
        float threshold);

    //=================
    /* Accumulators for circles */

    /*
    * Compute accumulator of a circle, with polar coordinates, from a
    * binary image.
    * Three parameters (x,y,r) so returns a matrix 3D.
    */
    cv::Mat accumulator_circle(cv::Mat image,
        int N_x, int N_y, int N_r,
        float r_min, float r_max);

    /*
    * Compute accumulator of a circle, with polar coordinates, from a list
    * of points.
    *
    */
    cv::Mat accumulator_circle_list(std::vector<std::pair<int,int>> list_of_points,
        int N_x, int N_y, int N_r,
        float r_min, float r_max,
        int H, int W);

    //=================
    /* Post processing*/

    /*
    * Search for segment bounds.
    */
    std::vector<Segment> find_line_bounds(std::vector<std::pair<float,float>> lines,
        cv::Mat image);

}
#endif
