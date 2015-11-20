#include <string>
#include <vector>
#include <iostream>
#include <locale>

#include "easyocr.h"
#include "webserver.h"
#include "helper.h"
#include "httpcurl.h"
#include "priv_config.h"

using namespace std;
using namespace cv;

class OcrController : public DynamicController {
public:
  virtual bool validPath(const char *path, const char *method) {
    if (strcmp(path, "/ocr") == 0 && strcmp("GET", method) == 0) {
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

    string imageSavePathRoot = PrivConfig::varBasePath;
    string imageSaveFullPath =
        Helper::GetHashPath(imageSavePathRoot, imgurl);
    eo.SetImageSavePathVIN(imageSavePathRoot + "/" + imageSaveFullPath + "_VIN.jpg");
    eo.SetImageSavePathRegDate(imageSavePathRoot + "/" + imageSaveFullPath + "_regDate.jpg");

    if (eo.Ocr()) {
      string result = eo.GetResult();
      response << "{" 
               << "\"status\": \"success\"" << "," 
               << "\"msg\": \"" << result << "\"" << "," 
               << "\"imgpath_VIN\": \"" << "/" << imageSaveFullPath << "_VIN.jpg" << "\"" << "," 
               << "\"imgpath_RegDate\": \"" << "/" << imageSaveFullPath << "_regDate.jpg" << "\"" 
               << "}";
    } else {
      response << "{\"status\": \"fail\", \"msg\": \"ocr fail\"}";
    }
    return;
  }
};
