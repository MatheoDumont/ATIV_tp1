#include "contour.h"
#include "kernel.h"

#include <cmath>

cv::Mat Contour::affinage_max_loc(cv::Mat in, cv::Mat pente, cv::Mat gradients)
{
	cv::Mat out = cv::Mat::zeros(in.rows, in.cols, CV_32F);

	for (int row = 1; row < in.rows - 1; row++)
	{
		for (int col = 1; col < in.cols - 1; col++)
		{

			if (in.at<float>(row, col) < 0.5f)
				continue;
			int steep = (int)pente.at<float>(row, col);
			int inv_steep = (steep + 4) % 8;

			std::pair<int, int> p_steep = Kernel::vec_from_direction(steep);
			p_steep = {p_steep.first + row, p_steep.second + col};

			std::pair<int, int> p_inv_steep = Kernel::vec_from_direction(inv_steep);
			p_inv_steep = {p_inv_steep.first + row, p_inv_steep.second + col};

			// 0 pixel dir grad, 1 pixel central, 2 pixel dir inv grad

			float val[3] = {
				abs(gradients.at<float>(p_steep.first, p_steep.second)),
				abs(gradients.at<float>(row, col)),
				abs(gradients.at<float>(p_inv_steep.first, p_inv_steep.second))};

			// pour simplifier
			std::pair<int, int> pos[3] = {p_steep, {row, col}, p_inv_steep};

			// argmax des gradients
			int imax = 0;
			float v_max = val[imax];

			for (int i = 1; i < 3; i++)
			{
				if (val[i] > v_max)
				{
					imax = i;
					v_max = val[imax];
				}
			}
			out.at<float>(pos[imax].first, pos[imax].second) = 1.f;
		}
	}
	return out;
}

cv::Mat Contour::fermeture_dil_ero(cv::Mat in,
								   std::vector<std::pair<int, int>> mask1,
								   std::vector<std::pair<int, int>> mask2,
								   int nb_it1, int nb_it2)
{
	cv::Mat out;
	in.copyTo(out);
	for (size_t i = 0; i < nb_it1; i++)
	{
		out = Contour::dilatation(out, mask1);
	}
	for (size_t i = 0; i < nb_it2; i++)
	{
		out = Contour::erosion(out, mask2);
	}
	return out;
}

cv::Mat Contour::fermeture_dil_dil(cv::Mat in,
								   std::vector<std::pair<int, int>> mask1, std::vector<std::pair<int, int>> mask2,
								   int nb_it1, int nb_it2)
{
	cv::Mat out;
	in.copyTo(out);
	for (size_t i = 0; i < nb_it1; i++)
	{
		out = Contour::dilatation(out, mask1, true);
	}
	for (size_t i = 0; i < nb_it2; i++)
	{
		out = Contour::dilatation(out, mask2, false);
	}
	return out;
}

cv::Mat Contour::fermeture_dilcont_affinage(cv::Mat in, cv::Mat pente, cv::Mat gradients,
											int nb_it1, int nb_it2)
{
	cv::Mat out;
	in.copyTo(out);
	for (size_t i = 0; i < nb_it1; i++)
	{
		out = Contour::dilatation_contour(out, pente, 3);
	}
	for (size_t i = 0; i < nb_it2; i++)
	{
		out = Contour::affinage_max_loc(out, pente, gradients);
	}
	return out;
}

cv::Mat Contour::dilatation(cv::Mat in, std::vector<std::pair<int, int>> mask,
							bool onContour)
{
	int i;
	int radius = 0;
	for (i = 0; i < mask.size(); i++)
	{
		radius = std::max(radius, std::max(abs(mask[i].first), abs(mask[i].second)));
	} // radius is now the maximum value of mask in order to deal with the borders

	cv::Mat out;
	if (onContour)
	{
		out = cv::Mat::zeros(in.rows, in.cols, CV_32F);
		for (int row = radius; row < in.rows - radius; row++)
		{
			for (int col = radius; col < in.cols - radius; col++)
			{
				if (in.at<float>(row, col) > 0.5)
				{ // it is a contour
					for (i = 0; i < mask.size(); i++)
					{
						out.at<float>(row + mask[i].first, col + mask[i].second) = 1.f;
					}
				}
			}
		}
	}
	else
	{
		out = cv::Mat::ones(in.rows, in.cols, CV_32F);
		for (int row = radius; row < in.rows - radius; row++)
		{
			for (int col = radius; col < in.cols - radius; col++)
			{
				if (in.at<float>(row, col) < 0.5)
				{ // it is not a contour
					for (i = 0; i < mask.size(); i++)
					{
						out.at<float>(row + mask[i].first, col + mask[i].second) = 0.f;
					}
				}
			}
		}
	}
	return out;
}

cv::Mat Contour::erosion(cv::Mat in, std::vector<std::pair<int, int>> mask)
{
	int i;
	int radius = 0;
	for (i = 0; i < mask.size(); i++)
	{
		radius = std::max(radius, std::max(abs(mask[i].first), abs(mask[i].second)));
	} // radius is now the maximum value of mask in order to deal with the borders

	bool toKeep = true;
	cv::Mat out = cv::Mat::zeros(in.rows, in.cols, CV_32F);
	for (int row = radius; row < in.rows - radius; row++)
	{
		for (int col = radius; col < in.cols - radius; col++)
		{
			toKeep = true;
			for (i = 0; i < mask.size(); i++)
			{
				if (in.at<float>(row + mask[i].first, col + mask[i].second) < 0.5)
				{
					toKeep = false;
					break;
				}
			}
			if (toKeep)
				out.at<float>(row, col) = 1.f;
		}
	}
	return out;
}

cv::Mat Contour::dilatation_contour(cv::Mat in, cv::Mat pente, int radius)
{
	cv::Mat out = cv::Mat::zeros(in.rows, in.cols, CV_32F);

	for (int row = radius; row < in.rows - radius; row++)
	{
		for (int col = radius; col < in.cols - radius; col++)
		{
			if (in.at<float>(row, col) > 0.5)
			{
				// it is a contour
				// on calcul le mask local dans la dir du gradient
				int steep = (int)pente.at<float>(row, col);
				int dir_contour = (steep + 2) % 8;
				std::pair<int, int> dir_steep = Kernel::vec_from_direction(dir_contour);

				// int dir_inv_contour = (dir_contour + 4) % 8;

				std::vector<std::pair<int, int>> mask(radius * 2 + 1);
				for (int i = -radius; i < radius + 1; i++)
				{
					if (i == 0)
						mask.push_back({0, 0});
					else
						mask.push_back({dir_steep.first * i, dir_steep.second * i});
				}
				// std::pair<int, int> mask[3] =
				// 	{Kernel::vec_from_direction(dir_contour), {0, 0}, Kernel::vec_from_direction(dir_inv_contour)};

				for (int i = 0; i < mask.size(); i++)
				{
					out.at<float>(row + mask[i].first, col + mask[i].second) = 1.f;
				}
			}
		}
	}
	return out;
}
