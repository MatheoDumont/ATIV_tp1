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
			std::vector<std::pair<int, int>> mask1, std::vector<std::pair<int, int>> mask2,
			int nb_it1 = 1, int nb_it2 = 1);

		/*
		 * Fermeture de contour par dilatation des contours puis dilatation des
		 * non-contour.
		 */
    static cv::Mat fermeture_dil_dil(cv::Mat in,
			std::vector<std::pair<int, int>> mask1, std::vector<std::pair<int, int>> mask2,
			int nb_it1 = 1, int nb_it2 = 1);

		/*
		 * Fermeture de contour par dilatation avec mask adaptatif dans le sens
		 * des contours (de rayon 1) puis affinage des contours.
		 * non-contour.
		 */
		static cv::Mat fermeture_dilcont_affinage(cv::Mat in, cv::Mat pente, cv::Mat gradients,
			int nb_it1, int nb_it2);

	 	static cv::Mat dilatation(cv::Mat in, std::vector<std::pair<int, int>> mask,
			 bool onContour = true);

		static cv::Mat erosion(cv::Mat in, std::vector<std::pair<int, int>> mask);

    /*
     * dilatation locale, dans le sens du contour
     */
    static cv::Mat dilatation_contour(cv::Mat in, cv::Mat pente, int radius);

};


#endif
