#include <string>
#include <vector>
#include <iostream>
#include <locale>

#include <boost/filesystem.hpp>

#include "easyocr.h"
#include "webserver.h"
#include "helper.h"
#include "httpcurl.h"
#include "priv_config.h"

using namespace std;

class DeleteFileController : public DynamicController {
public:
  virtual bool validPath(const char *path, const char *method) {
    if (strcmp(path, "/delete_file") == 0 && strcmp("GET", method) == 0) {
      return true;
    }
    return false;
  }

  virtual void createResponse(struct MHD_Connection *connection,
                              const char *url, const char *method,
                              const char *upload_data, size_t *upload_data_size,
                              std::stringstream &response) {

    const char *path = MHD_lookup_connection_value(
        connection, MHD_GET_ARGUMENT_KIND, "path");
    if (!path || !strstr(path, "/")) {
      response << "{\"status\": \"fail\", \"msg\": \"not a vaild path\"}";
      return;
    }

    string imageSavePathRoot = PrivConfig::varBasePath;
    boost::filesystem::path p(imageSavePathRoot + "/" + path);

    if (boost::filesystem::exists(p)) {
      if (boost::filesystem::remove(p))
        response << "{\"status\": \"success\", \"msg\": \"deleted\"}";
      else
        response << "{\"status\": \"fail\", \"msg\": \"delete failed\"}";
    } else {
      response << "{\"status\": \"fail\", \"msg\": \"file not exists\"}";
    }
    return;
  }
};
