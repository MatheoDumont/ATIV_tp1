#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>

/*
* Partie 1 Tp 1 Appliquer convolution sur Image
*/

int main()
{
    cv::Mat image = cv::imread("../datas/square_sample.png");

    if (!image.data)
    {
        std::cerr << "Error : image or video not found" << std::endl;
        exit(-1);
    }

    cv::imshow("Image_from_path", image);

    cv::waitKey(0);
    cv::destroyAllWindows();
}