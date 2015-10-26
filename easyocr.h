#ifndef __EASY_OCR_H__
#define __EASY_OCR_H__

#include <string>
#include <vector>
#include <ml.h>

class EasyOcr {
public:
  EasyOcr(std::vector<char> &);
  ~EasyOcr(void);

  std::string result;
  std::string roiImageSavePath;
  std::string GetResult(void);
  static bool LoadSVM(void);
  void SetROIImageSavePath(std::string);
  bool Ocr(void);

private:
  static CvSVM svmvin;
  static CvSVM svm33;
  static CvSVM svm35;
  static CvSVM svmLast5;
  cv::Mat mat;

  void WriteToDisk(std::string);
  void Resize(int maxWidth);
  bool TiltCorrect(void);
  bool Rotate(double angle);
  bool IfRotateNeed();
};

#endif
