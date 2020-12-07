#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include "kernel.h"
#include "seuil.h"
#include "path_contour.h"
#include "contour.h"

#define IMAGE_NAME0 "datas/square_sample.png"
#define IMAGE_NAME1 "datas/Palpa1.jpg"
#define IMAGE_NAME2 "datas/Palpa2.jpg"
#define IMAGE_NAME3 "datas/mr_piuel.jpeg"
#define IMAGE_NAME4 "datas/Lenna.png"
#define IMAGE_NAME5 "datas/circle_sample_0.png"
#define IMAGE_NAME6 "datas/circle_sample_1.png"
#define IMAGE_NAME7 "datas/circle_sample_2.png"
#include "hough_line.h"
#include "hough_cercle.h"

int main(int argc, char *argv[])
{
    cv::Mat horizontal = (cv::Mat_<float>(3, 3) << -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f, -1 / 3.f, 0.f, 1 / 3.f);
    cv::Mat vertical = (cv::Mat_<float>(3, 3) << 1 / 3.f, 1 / 3.f, 1 / 3.f, 0.f, 0.f, 0.f, -1 / 3.f, -1 / 3.f, -1 / 3.f);
    cv::Mat quart_plus = (cv::Mat_<float>(3, 3) << 1.f / 3.f, 1.f / 3.f, 0.f, 1.f / 3.f, 0.f, -1.f / 3.f, 0.f, -1.f / 3.f, -1.f / 3.f);
    cv::Mat quart_moins = (cv::Mat_<float>(3, 3) << 0.f, 1.f / 3.f, 1.f / 3.f, -1.f / 3.f, 0.f, 1.f / 3.f, -1.f / 3.f, -1.f / 3.f, 0.f);

    std::vector<cv::Mat> gradient_filters;
    gradient_filters.push_back(horizontal);
    gradient_filters.push_back(quart_moins);
    gradient_filters.push_back(vertical);
    gradient_filters.push_back(quart_plus);

    cv::Mat greyscale_image;
    cv::Mat im;
    std::string im_name = IMAGE_NAME6;
    bool line_detect = true; bool cercle_detect = false;
    if (argc > 1)
    {
        im_name = std::string(argv[1]);
        for (int i = 2 ; i < argc ; i++)
        {
            if (std::string(argv[i]) == "-lc")
                cercle_detect = true;
            else if (std::string(argv[i]) == "-c")
            {
                cercle_detect = true;
                line_detect = false;
            }
        }
    }
    else
    {
        std::cout << "------------------\n usage :    ./TP2 image_name -lc -c\n";
        std::cout << "-lc : for line+cercle detection\n";
        std::cout << "-c : for cercle only detection\n";
        std::cout << "by default, use line detection only\n-------------------\n";
    }

    std::cout << "image used : " << im_name << "\n";
    im = cv::imread(im_name.c_str());
    cv::cvtColor(im, greyscale_image, cv::COLOR_BGR2GRAY);
    greyscale_image.convertTo(greyscale_image, CV_32F);

    cv::Mat amp0;
    cv::Mat dir;

    std::vector<cv::Mat> gradient_convol = Kernel::conv2(greyscale_image, gradient_filters);

    amp0 = Kernel::amplitude_0(gradient_convol);
    dir = Kernel::angle(gradient_convol);

    cv::Mat im_threshold = Seuil::seuil_global(amp0 * (1 / 255.0), 0.1);


    if (line_detect)
    {
        std::cout << "LINE DETECTION\n-------------\n";
        HoughLine houghline(im_threshold, 200, 200);
        houghline.compute_accumulator();
        cv::Mat acc = houghline.get_accumulator();
        std::cout << "Accumulator : size = " << acc.rows << "x" << acc.cols << "\n";
        std::cout << "imthreshold : size = " << im_threshold.rows << "x" << im_threshold.cols << "\n";

        std::vector<Line_paremeters> lines = houghline.vote_threshold_local_maxima(0.0006, 8);
        for (int i = 0; i < lines.size(); i++)
        {
            std::cout << "polar paramaters of line " << i << " : (" << lines[i].first << ", " << lines[i].second << ")\n";
        }
        cv::Mat display0 = houghline.line_display_image_color(lines);
        cv::Mat display1 = houghline.segment_display_image(lines);

        cv::imshow("Lines in im", display0);
        cv::imshow("Segments in im", display1);
        cv::imshow("accumulator", acc * 2000.);
        cv::imshow("im thresh", im_threshold);

        cv::imwrite("output_lines.jpg", display0* 255.);
        cv::imwrite("im_contour_line.jpg", im_threshold* 255.);
        cv::imwrite("accumulator.jpg", acc*2000.* 255.);
        cv::waitKey(0);
    }

    if (cercle_detect)
    {
        std::cout << "CERCLE DETECTION\n-------------\n";
        HoughCercle houghcercle(im_threshold, 5.f, 0.85*std::min(im_threshold.cols,im_threshold.rows), 50, 50, 35);
        houghcercle.compute_accumulator();
        //cv::imshow("accumulator", houghcercle.accumulator * 2000); impossible to visalize the accumulator haha
        //cv::waitKey(0);
        std::cout << "accumulator size : "<< houghcercle.accumulator.size << "\n";
        // for (int ix = 0; ix < houghcercle.accumulator.size[0] - 0; ix++)
        // {
        //     for (int iy = 0; iy < houghcercle.accumulator.size[1] - 0; iy++)
        //     {
        //         for (int ir = 0; ir < houghcercle.accumulator.size[2] - 0; ir++)
        //         {
        //             std::cout << houghcercle.accumulator.at<float>(ix,iy,ir) << "\t";
        //         }
        //     }
        // }

        std::vector<Cercle_parameters> cercles = houghcercle.vote_threshold_local_maxima(0.00008, 5);
        std::cout << "cercles.size() : " << cercles.size() << "\n";
        for (int i = 0; i < cercles.size(); i++)
        {
            std::cout << "Cercle paramaters " << i << " : (" << std::get<0>(cercles[i]) << ", " << std::get<1>(cercles[i]) << ", " << std::get<2>(cercles[i]) << ")\n";
        }

        //cv::Mat display1 = houghcercle.cercle_display_image(cercles);
        cv::Mat display1 = houghcercle.cercle_display_image_color(im_threshold, cercles);
        cv::imshow("cercle in im0", display1);
        cv::imshow("im thresh", im_threshold);

        cv::imwrite("output_cercles.jpg", display1* 255.);
        cv::imwrite("im_contour.jpg", im_threshold* 255.);
        cv::waitKey(0);
    }
    return 0;
}
