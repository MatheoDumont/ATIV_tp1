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
     * Return the direction angle in [0,7] (and -1 if row=col=0) of the vector
		 * (row,col).
		 		3	2	1
 				4	x	0
 				5	6	7
     */
		static int direction_from_vec(int row, int col);


		/*
     * Return the vector of the neighboured position of (row,col) in
		 * direction d, taking into account the borders.
     */
		static std::vector<std::pair<int, int>> direction_neighbours(int rows,
			int cols, int row, int col, int d);

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
