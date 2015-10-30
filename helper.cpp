#include <iostream>
#include <sstream>
#include <string>
#include <locale>
#include <boost/filesystem.hpp>
#include <openssl/sha.h>
#include <string.h>
#include <stdio.h>

#include "helper.h"

std::string Helper::GetHashPath(std::string basePath, std::string filename) {
  std::string hashFilename = Helper::GetSHA1String(filename);
  std::string hashPath =
      hashFilename.substr(0, 3) + "/" + hashFilename.substr(3, 3);

  boost::filesystem::path p(basePath + "/" + hashPath);
  boost::filesystem::create_directories(p);

  return hashPath + "/" + hashFilename;
}

std::string Helper::GetSHA1String(std::string source) {
  unsigned char obuf[20];

  SHA1((unsigned char *)source.c_str(), source.size(), obuf);

  std::string output;
  int i;
  for (i = 0; i < 20; i++) {
    char tmpchar[5];
    sprintf(tmpchar, "%02x", obuf[i]);
    output.append(tmpchar);
  }
  return output;
}
