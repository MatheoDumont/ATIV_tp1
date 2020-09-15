
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/video/background_segm.hpp>
#include <iostream>
#include "kernel.h"

/*
* Partie 1 Tp 1 Appliquer convolution sur Image
*/

int main()
{
    float f[3][3] = {1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f, 1 / 3.f, 0.f, -1 / 3.f};
    float gauche[3][3] = {1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f, 1 / 9.f};
    
    cv::Mat filtre = cv::Mat(3, 3, CV_32F, &f);

    // std::cout << filtre << std::endl;

    std::vector<cv::Mat> filtres;
    filtres.push_back(filtre);

    // cv::Mat image = cv::imread("datas/square_sample.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image = cv::imread("datas/square_sample.png");

    cv::Mat gs;

    cv::cvtColor(image, gs, cv::COLOR_BGR2GRAY);

    std::cout << cv::typeToString(gs.type()) << std::endl;
    std::cout << cv::typeToString(filtre.type()) << std::endl;

    // return 0;
    if (!image.data)
    {
        std::cerr << "Error : image or video not found" << std::endl;
        exit(-1);
    }

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

    cv::destroyAllWindows();
}
