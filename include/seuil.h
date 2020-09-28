#ifndef SEUIL_H
#define SEUIL_H

#include <opencv4/opencv2/core.hpp>
#include "kernel.h"
#include <iostream>

class Seuil
{
private:
    Seuil(/* args */);
    ~Seuil();

public:
    static std::pair<int, int> centre_voisinage(int rows, int cols,
                                                int row, int col, int radius);

    static cv::Mat seuil_global(cv::Mat amp, float seuil);
    static cv::Mat seuil_local(cv::Mat amp, int taille_filtre, int k);

    /*
    * http://www.tsi.enst.fr/pages/enseignement/ressources/beti/hysteresis/principe.html
    */
    static cv::Mat seuil_hysteresis(cv::Mat amp, float seuil_low, float seuil_high, int radius_voisinage);
};

#endif
