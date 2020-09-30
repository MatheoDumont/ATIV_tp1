/* path_contour.cpp
 * Author : Yann-Situ GAZULL
 * Description :
 */

#include "path_contour.h"
#define NOT01(x) ((x)>0.25 && (x)<0.75)


int Path::direction_from_vec(int row, int col)
{
		// be careful : we use visual angles :
		/*  3	2	1
				4	x	0
				5	6	7 */
	if (row == 0 && col >  0)
		return 0;
	if (row >  0 && col >  0)
		return 1;
	if (row >  0 && col == 0)
		return 2;
	if (row >  0 && col <  0)
		return 3;
	if (row == 0 && col <  0)
		return 4;
	if (row <  0 && col <  0)
		return 5;
	if (row <  0 && col == 0)
		return 6;
	if (row <  0 && col >  0)
		return 7;
	return -1; // case (0,0)
}

std::vector<std::pair<int, int>> Path::direction_neighbours(int rows,
	int cols, int row, int col, int d)
{
	if (1 > col) // problem L
	{
		if (1 > row) // pb LU
		{
			return Path::direction_neighbours(rows,	cols, 1, 1, d);
		}
		else if (rows-2 < row) // pb LD
		{
			return Path::direction_neighbours(rows,	cols, rows-2, 1, d);
		}
		else // PB L bordures
		{
			return Path::direction_neighbours(rows,	cols, row, 1, d);
		}
	}
	else if (cols-2 < col) // problem R
	{
		if (1 > row) // pb RU
		{
			return Path::direction_neighbours(rows,	cols, 1, cols-2, d);
		}
		else if (rows-2 < row) // pb RD
		{
			return Path::direction_neighbours(rows,	cols, rows-2, cols-2, d);
		}
		else // PB R bordures
		{
			return Path::direction_neighbours(rows,	cols, row , cols-2, d);
		}
	}
	else if (1 > row) // pb U bordures
	{
		return Path::direction_neighbours(rows,	cols, 1, col, d);
	}
	else if (rows-2 < row) // pb D bordures
	{
		return Path::direction_neighbours(rows,	cols, rows-2, col, d);
	}
	else // cas général pas de pb
	{
		std::vector<std::pair<int, int>> res;
		// be careful : we use visual angles :
		/*  3	2	1
        4	x	0
		 		5	6	7 */
		if (d != 0 && d != 1 && d != 2)
			res.push_back({row+1,col-1});
		if (d != 1 && d != 2 && d != 3)
			res.push_back({row-1,col});
		if (d != 2 && d != 3 && d != 4)
			res.push_back({row-1,col+1});
		if (d != 3 && d != 4 && d != 5)
			res.push_back({row,col+1});
		if (d != 4 && d != 5 && d != 6)
			res.push_back({row+1,col+1});
		if (d != 5 && d != 6 && d != 7)
			res.push_back({row+1,col});
		if (d != 6 && d != 7 && d != 0)
			res.push_back({row+1,col-1});
		if (d != 7 && d != 0 && d != 1)
			res.push_back({row,col-1});
		// notice that if d not in [0,7], all the points are added.
		return res;
	}
}

cv::Mat Path::path_contour(cv::Mat amp, float seuil_low, float seuil_high)
{
	cv::Mat contour = cv::Mat::zeros(amp.rows, amp.cols, CV_32F);
  for(int row = 0; row < amp.rows; ++row) {
    for (int col = 0; col < amp.cols; ++col) {
			contour.at<float>(row,col) = 0.5;
			// O.5 means not decided yet if it is a contour or not
			// 0.0 means not a contour at call
			// 1.0 means a contour
		}
	}

	// create paths by calling path_contour
	for(int row = 0; row < amp.rows; ++row) {
    for (int col = 0; col < amp.cols; ++col) {
			if NOT01(contour.at<float>(row,col))
			{
				if (amp.at<float>(row,col) > seuil_high)
				{// a contour not already founded : let start a path !
					contour.at<float>(row,col) = 1.f;
					Path::path_from_pix(amp, row, col, -1 /* no direction */,
		 			 contour, seuil_low, seuil_high);
				}
				else if (amp.at<float>(row,col) < seuil_low)
				{// not a contour at all
					contour.at<float>(row,col) = 0.f;
				}
			}
		}
	}

	/* finally set remaining potential contour (between seuil_high and seuil_low)
	 * to not contour. */

	// for the moment to debug and see the behaviour, we set them to 0.2
	for(int row = 0; row < amp.rows; ++row) {
    for (int col = 0; col < amp.cols; ++col) {
			if NOT01(contour.at<float>(row,col))
				contour.at<float>(row,col) = 0.2;
		}
	}

	return contour;

}

void Path::path_from_pix(cv::Mat amp, int row, int col, int direction,
	cv::Mat &contour, float seuil_low, float seuil_high)
{
	contour.at<float>(row,col) = 1.f;
	/* start from a contour, that should have been chosen and already set to 1.f*/

	std::vector<std::pair<int, int>> n_pix =
				Path::direction_neighbours(amp.rows,	amp.cols, row, col, direction);
	/* The potential next pixels of the contour */

	std::vector<std::pair<int, int>> highs;
	std::pair<int, int> argmax;
	float maxi = 0.f;
	/* highs should contain pixels which amp > seuil_high
	 * argmax should contain pixel which amp is maximal but < seuil_high */

	 /*
	 * The idea is to create path on pixels that are in highs, and if highs's
	 * empty and amp[argmax] > seuil_low, continue this path from the argmax,
	 * otherwise stop the path.
	 */

	 for (size_t i = 0; i < n_pix.size(); i++)
	 {// for each pixel in n_pix :
	 		if NOT01(contour.at<float>(n_pix[i].first,n_pix[i].second))
			{// if not yet decided if it is a contour or not
				float a = amp.at<float>(n_pix[i].first,n_pix[i].second);
				if (a > seuil_high)	{
					contour.at<float>(n_pix[i].first,n_pix[i].second)=1.f;
					highs.push_back(n_pix[i]);
				}
				else	{
					if (a > maxi)	{
						maxi = a;
						argmax = n_pix[i];
					}
					contour.at<float>(n_pix[i].first,n_pix[i].second)=0.f;
					/* we set to 0.f. If we continue the pass on the max then we will
					 * change this choice on the argmax. */
				}
			}
	 }

	 if (highs.size() > 0)
	 {
		 	for (size_t j = 0; j < highs.size(); j++)
		 	{
				Path::path_from_pix(amp, highs[j].first, highs[j].second,
		 			Path::direction_from_vec(highs[j].first-row, highs[j].second-col),
					contour, seuil_low, seuil_high);

				// recursive calls on highs
			}
	 }
	 else if (maxi > seuil_low)
	 {
		 contour.at<float>(argmax.first, argmax.second)=1.f;
		 Path::path_from_pix(amp, argmax.first, argmax.second,
			 Path::direction_from_vec(argmax.first-row, argmax.second-col),
			 contour, seuil_low, seuil_high);
	 }
	 // otherwise : end of the contour
}
