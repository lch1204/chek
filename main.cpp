#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void func (Mat src , Mat hsv, Scalar lower_red, Scalar upper_red)
{
    Mat mask_red;
    inRange(hsv, lower_red, upper_red, mask_red);


    Mat mask_black; // Ивыделяем черные области
    bitwise_not(mask_red, mask_black);

    Mat binary;
    threshold(mask_black, binary, 200, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));//объединение близко расположенных точек
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // Поиск контуров черных областей
    std::vector<std::vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<Point>> filteredContours;
    for (const auto& contour : contours)
    {
        double epsilon = 0.03 * arcLength(contour, true);
        std::vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);

        if (approx.size() == 4 && isContourConvex(approx))
        {
            Rect rect = boundingRect(approx);
            double aspect_ratio = static_cast<double>(rect.width) / rect.height;
            if (rect.area() > 100 && aspect_ratio > 0.1 && aspect_ratio < 10)
            {
                if (rect.tl().x > 10 && rect.tl().y > 10 && rect.br().x < src.cols - 10 && rect.br().y < src.rows - 10)
                {
                    filteredContours.push_back(approx);
                }
            }
        }
    }

    for (size_t i = 0; i < filteredContours.size(); ++i)
    {
        drawContours(src, filteredContours, static_cast<int>(i), Scalar(0, 0, 255), 2);

        Rect roi_rect = boundingRect(filteredContours[i]);
        rectangle(src, roi_rect, Scalar(0, 255, 0), 2);

        Mat roi = src(roi_rect).clone();
        string window_name = "Region " + to_string(i + 1);
        namedWindow(window_name, WINDOW_NORMAL);
        imshow(window_name, roi);
    }

    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Result", src);
    waitKey(0);
}
int main(int argc, char** argv)
{
    // Загружаем изображение
    String imageName("../chek/chek3.jpg");
    Mat src = imread(imageName, IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "Не удалось открыть или найти изображение: " << imageName << endl;
        return -1;
    }

    // Преобразуем изображение в цветовое пространство HSV
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // Определяем нижний и верхний пороги для красного цвета
    Scalar lower_red(0, 40, 40);
    Scalar upper_red(15, 255, 255);
    func(src , hsv, lower_red, upper_red);

    String imageName2("../chek/chek5.jpg");
    Mat src2 = imread(imageName2, IMREAD_COLOR);
    if (src2.empty())
    {
        cerr << "Не удалось открыть или найти изображение: " << imageName2 << endl;
        return -1;
    }

    // Преобразуем изображение в цветовое пространство HSV
    Mat hsv2;
    cvtColor(src2, hsv2, COLOR_BGR2HSV);

    // Определяем нижний и верхний пороги для красного цвета
    Scalar lower_red2(0, 75, 60);
    Scalar upper_red2(13, 255, 255);
    func(src2 , hsv2, lower_red2, upper_red2);

    return 0;
}
