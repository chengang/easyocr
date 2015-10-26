#include <string>
#include <vector>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"

#include "easyocr.hpp"
#include "webcontroller.cpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

  cerr << "load svm data start.." << endl;
  EasyOcr::LoadSVM();
  cerr << "load svm data finished." << endl;

  MyController myController;
  WebServer server(81);
  server.addController(&myController);
  if (!server.start()) {
    cerr << "start server error" << endl;
    return 1;
  }

  cerr << "start server ok" << endl;
  return 0;
}
