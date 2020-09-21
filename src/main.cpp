
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/video/background_segm.hpp>
#include <iostream>
#include "kernel.h"
#include <math.h>
#include "seuil.h"

#define IMAGE_NAME "datas/square_sample.png"
/*
* Partie 1 Tp 1 Appliquer convolution sur Image
*/

int main()
{
  float f[3][3] = {1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f};

  cv::Mat filtre = cv::Mat(3, 3, CV_32F, &f);
  // std::cout << filtre << std::endl;
  std::vector<cv::Mat> filtres;
  filtres.push_back(filtre);

  // cv::Mat image = cv::imread("datas/square_sample.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat image = cv::imread(IMAGE_NAME);
  if (!image.data)
  {
    std::cerr << "Error : image or video not found" << std::endl;
    exit(-1);
  }

  cv::Mat gs; // for greyscale image

  cv::cvtColor(image, gs, cv::COLOR_BGR2GRAY);

  // std::cout << cv::typeToString(gs.type()) << std::endl;
  // std::cout << cv::typeToString(filtre.type()) << std::endl;

  cv::imshow("Image_from_path", image);
  cv::waitKey(0);

  cv::imshow("greyscale", gs);
  cv::waitKey(0);

  // cv::Mat gs_float;
  // gs.convertTo(gs_float, CV_32F);
  cv::Mat convol = Kernel::conv2(gs, filtres)[0];

  // cv::Mat convol_uchar;
  // convol.convertTo(convol_uchar, CV_8UC1);
  cv::imshow("convoluted", convol);
  cv::waitKey(0);

  // { // gradient tests with amplitude and angle
  //   std::vector<cv::Mat> gradient_filters;
  //   cv::Mat grad0 = (cv::Mat_<float>(3, 3) << 1.0/3, 0, -1.0/3, 1.0/3, 0, -1.0/3, 1.0/3, 0, -1.0/3);
  //   gradient_filters.push_back(grad0);
  //   cv::Mat grad2 = (cv::Mat_<float>(3, 3) << -1.0/3, -1.0/3, -1.0/3, 0, 0, 0, 1.0/3, 1.0/3, 1.0/3);
  //   gradient_filters.push_back(grad2);
  //
  //   std::vector<cv::Mat> gradient_convol = Kernel::conv2(gs, gradient_filters);
  //   cv::imshow("amplitude_0 grad", Kernel::amplitude_0(gradient_convol)*(1/255.0));
  //   cv::imshow("amplitude_1 grad", Kernel::amplitude_1(gradient_convol)*(1/255.0));
  //   cv::imshow("amplitude_2 grad", Kernel::amplitude_2(gradient_convol)*(1/255.0));
  //   cv::waitKey(0);
  //
  //   cv::Mat gradient_color = Kernel::color_gradient_im(Kernel::amplitude_0(gradient_convol),
	// 					       Kernel::angle_arctan(gradient_convol[0],
	// 							     gradient_convol[1]));
  //   cv::imshow("color gradient", gradient_color*(1/255.0));
  //   cv::waitKey(0);
  // }

  {
    // gradients 4 directions
    std::vector<cv::Mat> gradient_filters;
    cv::Mat gauche = (cv::Mat_<float>(3, 3) << 1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f);
    cv::Mat droite = (cv::Mat_<float>(3, 3) << -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f);
    cv::Mat quart_plus = (cv::Mat_<float>(3, 3) << 1.f/ 3.f, 1.f/ 3.f, 0.f, 1.f/ 3.f, 0.f, -1.f / 3.f, 0.f, -1.f/ 3.f ,-1.f/ 3.f);
    cv::Mat quart_moins = (cv::Mat_<float>(3, 3) << 0.f, 1.f/ 3.f, 1.f/ 3.f, -1.f/ 3.f, 0.f, 1.f/ 3.f, -1.f/ 3.f, -1.f/ 3.f, 0.f);

    gradient_filters.push_back(gauche);
    gradient_filters.push_back(droite);
    gradient_filters.push_back(quart_plus);
    gradient_filters.push_back(quart_moins);

    std::vector<cv::Mat> gradient_convol = Kernel::conv2(gs, gradient_filters);

    cv::Mat gradient_color = Kernel::color_gradient_im(
                      Kernel::amplitude_0(gradient_convol),
						          Kernel::angle(gradient_convol) * (M_PI/4)
    );
    cv::imshow("color gradient", gradient_color*(1/255.0));
    cv::waitKey(0);

    // TEST SEUIL GLOBAL
    cv::imshow("seuillage global 4 directions", Seuil::seuil_global(Kernel::amplitude_0(gradient_convol)));
    cv::waitKey(0);

  }

  cv::destroyAllWindows();
}
