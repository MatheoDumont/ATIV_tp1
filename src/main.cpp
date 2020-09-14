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
    cv::Mat image = cv::imread("../datas/square_sample.png",CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data)
    {
        std::cerr << "Error : image or video not found" << std::endl;
        exit(-1);
    }

    //Mat horiz_filter = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::imshow("Image_from_path", image);

    cv::waitKey(0);
    cv::destroyAllWindows();
}
