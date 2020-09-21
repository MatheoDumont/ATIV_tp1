#ifndef KERNEL_H
#define KERNEL_H

#include <opencv4/opencv2/core.hpp>

class Kernel
{
private:
    Kernel(/* args */);
    ~Kernel();

public:
    /*
     * Compute convolution on the image `in` with the filter `filtre` at the pixel (row, col)
     * and store the output in `out`.
     */
    static float conv_pixel(cv::Mat &in, int row, int col, cv::Mat filtre);

    /*
    * Compute the convolution of the filters `filtres` on the image `image`
    * and return `std::vector<cv::Mat>` that contains the resultat of the convolution
    * of each filter on the image.
    */
    static std::vector<cv::Mat> conv2(cv::Mat &image, std::vector<cv::Mat> filtres);

    /*
     * Compute amplitude of the multidirectional gradient mi =(m_0,m_1,...m_n)
     * with the L_x norm : pow(1/n pow(|m_i|,x),1/x).
     */
    static cv::Mat amplitude_x(std::vector<cv::Mat> mi, float x);

    /*
     * Compute amplitude of the multidirectional gradient mi =(m_0,m_1,...m_n)
     * with the L_infinity norm : max_{0\leq i\leq n} |m_i|.
     * Equivalent to `amplitude_x(mi, x->infinity)`.
     */
    static cv::Mat amplitude_0(std::vector<cv::Mat> mi);

    /*
     * Compute amplitude of the multidirectional gradient mi =(m_0,m_1,...m_n)
     * with the L_1 norm : 1/n \sum_i |m_i|.
     * Equivalent to `amplitude_x(mi, x=1.0)`.
     */
    static cv::Mat amplitude_1(std::vector<cv::Mat> mi);

    /*
     * Compute amplitude of the multidirectional gradient mi =(m_0,m_1,...m_n)
     * with the euclidean norm L_2 : sqrt(1/n \sum_i m_i^2)
     * Equivalent to `amplitude_x(mi, x=2.0)`.
     */
    static cv::Mat amplitude_2(std::vector<cv::Mat> mi);

    /*
     *  Compute angle of direction of the gradient (mx,my) with atan2
     * (of math.h).
     */
    static cv::Mat angle_arctan(cv::Mat mx, cv::Mat my);

    /*
     *  Compute angle indice in {0,1,...,2|mi|-1} by looking the gradient that have
     * the max amplitude value in |mi|.
		 * If the max is |mi[i]| and mi[i] > 0 then return i,
		 * if mi[i] < 0 return i + |mi|.
     */
    static cv::Mat angle(std::vector<cv::Mat> mi);

    static cv::Mat color_gradient_im(cv::Mat amp, cv::Mat ang);
};
void HSVtoRGB(float H, float S,float V, cv::Vec3f& pixel);
#endif
