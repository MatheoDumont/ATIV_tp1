#ifndef CONTOUR_H
#define CONTOUR_H

#include <opencv4/opencv2/core.hpp>

class Contour
{
private:
    Contour(/* args */);
    ~Contour();

public:

    /*
    * affinage : avec hard code le nombre de direction (impl variable global ?)
    * retourne l'image avec des contours affines en selectionner le pixel possedant le gradient max dans le voisinage.
    */
    static cv::Mat affinage_max_loc(cv::Mat in, cv::Mat pente, cv::Mat gradients);

		/*
		 * Fermeture de contour par grossissement-amincissement (dilatation erosion)
		 */
    static cv::Mat fermeture_dil_ero(cv::Mat in,
			std::vector<std::pair<int, int>> mask, int nb_iteration = 1);

		/*
		 * Fermeture de contour par dilatation des contours puis dilatation des
		 * non-contour.
		 */
    static cv::Mat fermeture_dil_dil(cv::Mat in,
			std::vector<std::pair<int, int>> mask, int nb_iteration = 1);

	 	static cv::Mat dilatation(cv::Mat in, std::vector<std::pair<int, int>> mask,
			 bool onContour = true);

		static cv::Mat erosion(cv::Mat in, std::vector<std::pair<int, int>> mask);

};


#endif
