#include <iostream>
#include <string>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"
#include "easyocr.h"
#include "opencv.cpp"

CvSVM EasyOcr::svmvin;
CvSVM EasyOcr::svm35;
CvSVM EasyOcr::svm33;
CvSVM EasyOcr::svmLast5;

EasyOcr::EasyOcr(std::vector<char> &imgData) {
  this->mat = imdecode(cv::Mat(imgData), CV_LOAD_IMAGE_COLOR);
  this->result = "";
}

EasyOcr::~EasyOcr(void) { this->mat.release(); }

bool EasyOcr::LoadSVM(void) {
  EasyOcr::svmvin.load("../data/SVM_D.xml");
  EasyOcr::svm35.load("../data/SVM_DATA_all0-Q_1000_LINEAR.xml");
  EasyOcr::svm33.load("../data/SVM_DATA_all0-Z_1000_LINEAR.xml");
  EasyOcr::svmLast5.load("../data/SVM_DATA_all0-9_1000_LINEAR.xml");
  return true;
}

std::string EasyOcr::GetResult(void) { return this->result; }

void EasyOcr::SetROIImageSavePath(std::string path) {
  this->roiImageSavePath = path;
}

bool EasyOcr::Ocr(void) {
  //this->Resize(720);
  //this->TiltCorrect();

  char pre[30] = "";
  bool ifRecSucc = recvin(this->mat, pre, &EasyOcr::svmvin, &EasyOcr::svm35, &EasyOcr::svm33, &EasyOcr::svmLast5, this->roiImageSavePath.c_str());
  this->result = pre;
  return ifRecSucc;
}

void EasyOcr::WriteToDisk(std::string filename) {
  imwrite(filename, this->mat);
}

void EasyOcr::Resize(int maxWidth) {
  double dar = (double)this->mat.rows / this->mat.cols;
  int newHeight = dar * maxWidth;
  cv::Mat matDst;
  cv::resize(this->mat, matDst, cv::Size(maxWidth, newHeight));
  this->mat = matDst;
}

bool EasyOcr::TiltCorrect(void) {
  if (this->IfRotateNeed()) {
    // std::cout << "need" << std::endl;
  } else {
    std::cout << "no need" << std::endl;
  }
  this->Rotate(90);
  return true;
}

bool EasyOcr::Rotate(double angle) {
  cv::Mat matDst;
  cv::Point center = cv::Point(this->mat.cols / 2, this->mat.rows / 2);
  cv::Mat matRotate = cv::getRotationMatrix2D(center, angle, 1);

  cv::Rect bbox =
      cv::RotatedRect(center, this->mat.size(), angle).boundingRect();
  matRotate.at<double>(0, 2) += bbox.width / 2 - center.x;
  matRotate.at<double>(1, 2) += bbox.height / 2 - center.y;

  cv::warpAffine(this->mat, matDst, matRotate, bbox.size());
  this->mat = matDst;

  return true;
}

bool EasyOcr::IfRotateNeed(void) {
  cv::Mat matDst;
  // find vertical lines
  cv::Sobel(this->mat, matDst, this->mat.depth(), 1,
            0); // 1,0-Vertical 0,1-Horizon

  // rgb to gray
  cv::cvtColor(matDst, matDst, CV_BGR2GRAY);
  cv::threshold(matDst, matDst, 0, 255,
                cv::THRESH_OTSU |
                    cv::THRESH_BINARY); // cv::THRESH_OTSU?? not in document

  // be fatter
  cv::Mat kernelForDilate = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(1, 2), cv::Point(0, 0));
  cv::dilate(matDst, matDst, kernelForDilate, cv::Point(-1, -1), 6);

  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  findContours(matDst, contours, hierarchy, CV_RETR_EXTERNAL,
               CV_CHAIN_APPROX_SIMPLE);

  int countVerticalLine = 0;
  int maxY = 0;
  for (int i = 0; i < contours.size(); i++) {
    cv::Rect aRect = boundingRect(contours[i]);
    if (aRect.height > 100 &&
        abs(aRect.x - matDst.cols / 2) < matDst.cols / 8 && aRect.width < 30) {
      countVerticalLine++;
      maxY = cv::max(maxY, aRect.y + aRect.height);
    }
  }

  if (countVerticalLine < 3) {
    return false;
  }
  return true;
}
