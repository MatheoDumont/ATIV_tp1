/* seuil.cpp
 * Contains threshold functions.
 */
#include "seuil.h"

std::pair<int, int> Seuil::centre_voisinage(int rows, int cols,
											int row, int col, int radius)
{
	if (radius > col) // problem L
	{
		if (radius > row) // pb LU
		{
			return {radius, radius};
		}
		else if (rows - 1 - radius < row) // pb LD
		{
			return {rows - 1 - radius, radius};
		}
		else // PB L bordures
		{
			return {row, radius};
		}
	}
	else if (cols - 1 - radius < col) // problem R
	{
		if (radius > row) // pb RU
		{
			return {radius, cols - 1 - radius};
		}
		else if (rows - 1 - radius < row) // pb RD
		{
			return {rows - 1 - radius, cols - 1 - radius};
		}
		else // PB R bordures
		{
			return {row, cols - 1 - radius};
		}
	}
	else if (radius > row) // pb U bordures
	{
		return {radius, col};
	}
	else if (rows - 1 - radius < row) // pb D bordures
	{
		return {rows - 1 - radius, col};
	}
	else // cas général pas de pb
	{
		return {row, col};
	}
}

cv::Mat Seuil::seuil_global(cv::Mat amp, float seuil)
{
	//float mean = cv::mean(amp)[0];

	cv::Mat res = cv::Mat::zeros(amp.rows, amp.cols, CV_32F);
	for (int row = 0; row < amp.rows; row++)
	{
		for (int col = 0; col < amp.cols; col++)
		{
			if (amp.at<float>(row, col) < seuil)
				res.at<float>(row, col) = 0.f;
			else
				res.at<float>(row, col) = 1.f; //amp.at<float>(row, col);
		}
	}
	return res;
}

cv::Mat Seuil::seuil_local(cv::Mat amp, int taille_filtre, int k)
{
	float num = 1 / float(taille_filtre * taille_filtre);
	cv::Mat mean_filter = cv::Mat::zeros(taille_filtre, taille_filtre, CV_32F);
	// cv::Mat mean_filter = (cv::Mat_<float>(3, 3) << 1/9.f, 1/9.f, 1/9.f, 1/9.f, 1/9.f, 1/9.f,1/9.f,1/9.f,1/9.f);
	for (int row = 0; row < taille_filtre; row++)
	{
		for (int col = 0; col < taille_filtre; col++)
		{
			mean_filter.at<float>(row, col) = num;
		}
	}

	cv::Mat res = cv::Mat::zeros(amp.rows, amp.cols, CV_32F);
	for (int row = 0; row < amp.rows; ++row)
	{
		for (int col = 0; col < amp.cols; ++col)
		{

			// correction de la position pour les cas de bordure
			std::pair<int, int> corrected_position = centre_voisinage(amp.rows, amp.cols, row, col, (taille_filtre - 1) / 2);
			float local_mean = Kernel::conv_pixel(amp, corrected_position.first, corrected_position.second, mean_filter);
			// std::cout << local_mean << " " << amp.at<float>(row, col) << std::endl;

			if (amp.at<float>(row, col) < (local_mean * k))
				res.at<float>(row, col) = 0.f;
			else
				res.at<float>(row, col) = 1.f; //amp.at<float>(row, col);
		}
	}
	return res;
}

cv::Mat Seuil::seuil_hysteresis(cv::Mat amp, float seuil_low, float seuil_high, int radius_voisinage)
{
	cv::Mat seuil_glob = Seuil::seuil_global(amp, seuil_high);

	cv::Mat res = cv::Mat::zeros(amp.rows, amp.cols, CV_32F);
	for (int row = 0; row < amp.rows; ++row)
	{
		for (int col = 0; col < amp.cols; ++col)
		{
			if (seuil_glob.at<float>(row, col) > 0.5) // already a contour
				res.at<float>(row, col) = 1.0;
			else if (amp.at<float>(row, col) < seuil_low) // not a contour at all
				res.at<float>(row, col) = 0.0;
			else // maybe a contour : look if there is already a contour in its neighboorhood
			{
				std::pair<int, int> corrected_position = centre_voisinage(amp.rows,
																		  amp.cols, row, col, radius_voisinage);
				bool contour_found = false;
				for (int y = -radius_voisinage; y < radius_voisinage + 1; y++)
				{
					for (int x = -radius_voisinage; x < radius_voisinage + 1; x++)
					{
						if (seuil_glob.at<float>(corrected_position.first + y, corrected_position.second + x) > 0.5)
							contour_found = true;
						if (contour_found)
							break;
					}
					if (contour_found)
						break;
				}
				if (contour_found)
					res.at<float>(row, col) = 1.0;
				else
					res.at<float>(row, col) = 0.0;
			}
		}
	}
	return res;
}
