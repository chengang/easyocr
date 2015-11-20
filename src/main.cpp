#include <string>
#include <vector>
#include <iostream>

#include <cstdlib>

#include <boost/filesystem.hpp>
#include "opencv2/highgui/highgui.hpp"

#include "easyocr.h"
#include "controller_deletefile.cpp"
#include "controller_ocr.cpp"
#include "helper.h"
#include "priv_config.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  cerr << "EasyOCR version 0.1.1" << endl;

  long listenPort = 0;
  string varDirectory = "";
  if (argc >= 2 && Helper::isNumber(argv[1]) ) {
    listenPort = strtol(argv[1], NULL, 0);
  } else {
    cerr << "Please provide port number in the first argument." << endl;
    return 0;
  }

  if (argc >= 3 && boost::filesystem::is_directory(argv[2]) ) {
    varDirectory = argv[2];
  } else {
    cerr << "Please provide var directory in the second argument." << endl;
    return 0;
  }
  PrivConfig::varBasePath = varDirectory;

  cerr << "Load svm data start.." << endl;
  EasyOcr::LoadSVM();
  cerr << "Load svm data finished." << endl;

  cerr << "Server start at port " << listenPort << "." << endl;
  WebServer server(listenPort);
  OcrController ocrController;
  DeleteFileController deleteFileController;
  server.addController(&ocrController);
  server.addController(&deleteFileController);

  if (!server.start()) {
    cerr << "Server start error." << endl;
    return 1;
  }

  cerr << "Server stopped." << endl;
  return 0;
}
