/* seuil.cpp
 * Contains threshold functions.
 */
#include "seuil.h"

static std::pair<int,int> Seuil::centre_voisinage(int rows, int cols,
 																					int row, int col, int radius)
{

	if (radius > col) // problem U
	{
		if (radius > row) // pb UL
		{

		}
		elif (rows-1-radius < row) // pb UR
		{

		}
		else // PB U bordures
		{

		}

	}
	elif (cols-1-radius < col) // problem D
	{
		if (radius > row) // pb DL
		{

		}
		elif (rows-1-radius < row) // pb DR
		{

		}
		else // PB D bordures
		{

		}
	}
	elif (radius > row) // pb L bordures
	{

	}
	elif (rows-1-radius < row) // pb R bordures
	{

	}
	else // cas général pas de pb
	{

	}
}

static cv::Mat Seuil::seuil_global(cv::Mat amp)
{
  float mean = cv::mean(amp);
  cv::Mat res = cv::Mat::zeros(amp.rows, amp.cols, CV_32F);
  for (int row = 0; row < amp.rows; row++) {
    for (int col = 0; col < amp.cols; col++) {
      if (amp.at<float>(row, col) < mean)
        res.at<float>(row, col) = 0;
      else
        res.at<float>(row, col) = amp.at<float>(row, col);
    }
  }
  return res;
}

static cv::Mat Seuil::seuil_local(cv::Mat amp)
{

}

static cv::Mat Seuil::seuil_hysteresis(cv::Mat amp)
{

}
