/* path_contour.h
 * Description :
 */
#ifndef PATH_CONTOUR_H
#define PATH_CONTOUR_H

#include <opencv4/opencv2/core.hpp>
#include "kernel.h"
#include <iostream>

class Path
{
	private:
		Path();
		~Path();

	public:
		/*
     * Return the vector of the neighboured position of (row,col) in
		 * direction d, taking into account the borders.
     */
		static std::vector<std::pair<int, int>> direction_neighbours(int rows,
			int cols, int row, int col, int d, bool three_neighbourhood = true);

		/*
     * Return a binary matrix of contour with the path_contour method
     */
		static cv::Mat path_contour(cv::Mat amp, float seuil_low, float seuil_high);

		/*
     * Recursive function that follow a contour path from a pixel position
     */
		static void path_from_pix(cv::Mat amp, int row, int col, int direction,
			 cv::Mat &contour, float seuil_low, float seuil_high);

};

#endif
