#include <string>
#include <vector>
#include <iostream>
#include <locale>

#include "easyocr.h"
#include "webserver.h"
#include "helper.h"
#include "httpcurl.h"

using namespace std;
using namespace cv;

class MyController : public DynamicController {
public:
  virtual bool validPath(const char *path, const char *method) {
    if (strcmp(path, "/") == 0 && strcmp("GET", method) == 0) {
      return true;
    }
    return false;
  }

  virtual void createResponse(struct MHD_Connection *connection,
                              const char *url, const char *method,
                              const char *upload_data, size_t *upload_data_size,
                              std::stringstream &response) {

    const char *imgurl = MHD_lookup_connection_value(
        connection, MHD_GET_ARGUMENT_KIND, "imgurl");
    if (!imgurl || !strstr(imgurl, "http://")) {
      response << "{\"status\": \"fail\", \"msg\": \"not a http url\"}";
      return;
    }

    string curlurl(imgurl);
    vector<char> imgData;
    if (!HttpCurl::GetVector(curlurl, imgData)) {
      response << "{\"status\": \"fail\", \"msg\": \"download image fail\"}";
      return;
    }

    EasyOcr eo(imgData);

    string roiImageSavePathRoot = "/var/www";
    string roiImageSaveFullPath = Helper::GetHashPath(roiImageSavePathRoot, imgurl);
    eo.SetROIImageSavePath(roiImageSavePathRoot + roiImageSaveFullPath );
    if (eo.Ocr()) {
      string result = eo.GetResult();
      response << "{\"status\": \"success\", \"msg\": \"" << result << "\", \"imgpath\": \"" << roiImageSaveFullPath << "\"}";
    } else {
      response << "{\"status\": \"fail\", \"msg\": \"ocr fail\"}";
    }
    return;
  }
};
