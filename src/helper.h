#ifndef __HELPER_H__
#define __HELPER_H__

#include <string>
#include <locale>

class Helper {
public:
  static std::string GetHashPath(std::string, std::string);
  static bool isNumber(const std::string &);

private:
  static std::string GetSHA1String(std::string);
};

#endif
