#ifndef KERNEL_H
#define KERNEL_H

#include <opencv4/opencv2/core.hpp>

class Kernel
{
private:
    /* data */
public:
    Kernel(/* args */);
    ~Kernel();

    /*
     * Compute convolution on the image `in` with the filter `filtre` at the pixel (row, col)
     * and store the output in `out`.
     */
    void conv_pixel(cv::Mat &in, cv::Mat &out, int row, int col, cv::Mat filtre);

    /*
    * Compute the convolution of the filters `filtres` on the image `image`
    * and return `std::vector<cv::Mat>` that contains the resultat of the convolution
    * of each filter on the image.
    */
    std::vector<cv::Mat> conv2(cv::Mat &image, std::vector<cv::Mat> filtres);

    /*
     * Compute amplitude of the gradient (mx,my) with the euclidean norm :
     * sqrt(mx^2+my^2)/sqrt(2). We divide by sqrt(2) in order to have a result
     * between 0 and 255.
     */
    cv::Mat amplitude_0(cv::Mat mx, cv::Mat my);

    /*
     * Compute amplitude of the multidirectional gradient mi =(m_0,m_1,...m_n)
     * with the L_0 norm :
     * max_{0\leq i\leq n} |m_i|.
     */
    cv::Mat amplitude_1(std::vector<cv::Mat> mi); // max |D_i|

    /*
     *  Compute angle of direction of the gradient (mx,my) with atan2
     * (of math.h).
     */
    cv::Mat angle_0(cv::Mat mx, cv::Mat my);
};

Kernel::Kernel(/* args */)
{
}

Kernel::~Kernel()
{
}

#endif
