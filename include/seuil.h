#ifndef SEUIL_H
#define  SEUIL_H

#include <opencv4/opencv2/core.hpp>

class Seuil
{
  private:
      Seuil(/* args */);
      ~Seuil();

  public:
		static std::pair<int,int> centre_voisinage(int rows, int cols,
																							 int row, int col, int radius);
    static cv::Mat seuil_global(cv::Mat amp);
    static cv::Mat seuil_local(cv::Mat amp);
    static cv::Mat seuil_hysteresis(cv::Mat amp);
};

#endif
