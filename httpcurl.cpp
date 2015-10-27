#include <iostream>
#include "curl/curl.h"
#include "httpcurl.h"

size_t HttpCurl::OnWriteData(void *buf, size_t size, size_t nmemb,
                             void *userdata) {
  std::vector<char> *vec =
      dynamic_cast<std::vector<char> *>((std::vector<char> *)userdata);
  if (!vec || !buf) {
    return -1;
  }
  char *pData = (char *)buf;
  vec->insert(vec->end(), pData, pData + size * nmemb);
  return nmemb;
}

bool HttpCurl::GetVector(std::string &url, std::vector<char> &resp) {
  CURLcode res;
  CURL *curl = curl_easy_init();
  if (!curl)
    return false;

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 9L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 6L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 300L);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&resp);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, HttpCurl::OnWriteData);

  res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << curl_easy_strerror(res) << std::endl;
    return false;
  }

  long httpcode = 0;
  res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpcode);
  if (res != CURLE_OK || httpcode != 200) {
    std::cerr << "http code is not 200 (" << httpcode << ")" << std::endl;
    return false;
  }

  curl_easy_cleanup(curl);
  return true;
}
