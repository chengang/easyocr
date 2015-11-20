#ifndef __HTTP_CURL_H__
#define __HTTP_CURL_H__

#include <string>
#include <vector>

class HttpCurl {
public:
  static bool GetVector(std::string &, std::vector<char> &);

private:
  static size_t OnWriteData(void *, size_t, size_t, void *);
};

#endif
